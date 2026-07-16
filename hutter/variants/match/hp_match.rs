// hp_match.rs — dep-free lossless codec, Hutter pre-test variant key="match".
// SAME binary arithmetic coder (fpaq0-derived, proven-symmetric) as the base
// hutter_pilot.rs — Encoder/Decoder/flush/renorm COPIED VERBATIM, unchanged.
// Only the MODEL changed: order-2 direct context MIXED (logistic mixer) with an
// LZP match model (hash(last 4 bytes) -> last position; predict the byte that
// followed that position). Fully causal + byte-exact. Aimed at repetitive text.
//
// Modes (same CLI as base):
//   verify     <file>            : in-memory roundtrip + print sizes/bpc (asserts exact)
//   compress   <in> <archive>    : write archive (8-byte LE length header + AC stream)
//   decompress <archive> <out>   : reconstruct byte-exact
//
// Reports one machine row:  PILOT|mode=..|input_bytes=..|archive_bytes=..|
//                           decoder_src_bytes=..|total_bytes=..|ratio_x=..|bpc=..|roundtrip=..

use std::env;
use std::fs;
use std::process::exit;

const NCTX: usize = 1 << 24; // order-2 exact: (prev2<<8 | prev1)<<8 | node  (node 1..255)
const HTBITS: u32 = 22;
const HTSIZE: usize = 1 << 22; // hash table: hash(last 4 bytes) -> position
const MM_SIZE: usize = 1024;   // match-model prob table: (len_bucket<<4)|(bitpos<<1)|pred_bit
const MINLEN: usize = 4;       // context length hashed for LZP
const WCTX: usize = 16;        // mixer weight contexts (by match state)
const WBOUND: i32 = 1 << 20;   // mixer weight clamp

// squash(d): stretched -> probability, returns 0..4095 (lpaq/paq table).
#[inline]
fn squash(d: i32) -> i32 {
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
        3348, 3607, 3785, 3901, 3968, 4004, 4022, 4030, 4034, 4036, 4038, 4038, 4038, 4038,
    ];
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 0;
    }
    let w = d & 127;
    let idx = ((d >> 7) + 16) as usize;
    (T[idx] * (128 - w) + T[idx + 1] * w + 64) >> 7
}

// Model: order-2 direct context + LZP match model, combined by a 2-input logistic mixer.
struct Model {
    // order-2 direct context (identical to base)
    t: Vec<u16>,
    c1: u32,
    c2: u32,
    // match model
    mm: Vec<u16>,         // adaptive P(bit=1) per (len_bucket, bitpos, pred_bit)
    hist: Vec<u8>,        // causal history (input on encode, decoded output on decode)
    ht: Vec<u32>,         // hash(last 4 bytes) -> position of the byte that followed
    mptr: usize,          // predicted-source pointer into hist
    mlen: u32,            // verified consecutive-match length
    mvalid: bool,         // do we currently have a match prediction?
    match_byte: u8,       // hist[mptr] captured at byte start
    mm_active: bool,      // is the match still consistent with the partial byte?
    // mixer
    w: Vec<i32>,          // WCTX * 2 weights (16.16 fixed point)
    stretch_tab: Vec<i32>,
    // per-bit transient (recomputed each predict, consumed by update_bit)
    tx0: i32,
    tx1: i32,
    wctx: usize,
    mm_idx: usize,
    mm_used: bool,
    pr: u32,
}

impl Model {
    fn new() -> Self {
        // build stretch table (inverse of squash)
        let mut st = vec![0i32; 4096];
        let mut pi = 0usize;
        for x in -2047..=2047 {
            let i = squash(x) as usize;
            for j in pi..=i {
                st[j] = x;
            }
            pi = i + 1;
        }
        for j in pi..4096 {
            st[j] = 2047;
        }
        let mut w = vec![0i32; WCTX * 2];
        for c in 0..WCTX {
            w[c * 2] = 1 << 16; // order-2 input starts at weight 1.0
            w[c * 2 + 1] = 0;   // match input starts at 0.0
        }
        Model {
            t: vec![2048u16; NCTX],
            c1: 0,
            c2: 0,
            mm: vec![2048u16; MM_SIZE],
            hist: Vec::new(),
            ht: vec![u32::MAX; HTSIZE],
            mptr: 0,
            mlen: 0,
            mvalid: false,
            match_byte: 0,
            mm_active: false,
            w,
            stretch_tab: st,
            tx0: 0,
            tx1: 0,
            wctx: 0,
            mm_idx: 0,
            mm_used: false,
            pr: 2048,
        }
    }

    #[inline]
    fn idx(&self, node: u32) -> usize {
        ((((self.c2 << 8) | self.c1) << 8) | node) as usize & (NCTX - 1)
    }

    // Called once per byte, before coding its 8 bits.
    #[inline]
    fn begin_byte(&mut self) {
        if self.mvalid {
            self.match_byte = self.hist[self.mptr];
            self.mm_active = true;
        } else {
            self.match_byte = 0;
            self.mm_active = false;
        }
    }

    // Predict P(bit=1) for the given bit-tree node at shift i (7..0). Stores transient
    // state for update_bit. Does NOT mutate any learned table (deterministic on both sides).
    #[inline]
    fn predict(&mut self, node: u32, i: u32) -> u32 {
        // order-2 input
        let p0 = self.t[self.idx(node)] as usize; // 0..4095
        let st0 = self.stretch_tab[p0];
        // match input (only while the partial byte still agrees with match_byte)
        let mm_used = self.mm_active;
        let predicted_bit = if mm_used {
            ((self.match_byte >> i) & 1) as usize
        } else {
            0
        };
        let len_bucket = if self.mlen > 63 { 63 } else { self.mlen as usize };
        let mm_idx = (len_bucket << 4) | ((i as usize) << 1) | predicted_bit;
        let st1 = if mm_used {
            self.stretch_tab[self.mm[mm_idx] as usize]
        } else {
            0
        };
        // mixer weight context by match state
        let wctx = if self.mm_active {
            1 + std::cmp::min(self.mlen as usize, WCTX - 2)
        } else {
            0
        };
        let w0 = self.w[wctx * 2] as i64;
        let w1 = self.w[wctx * 2 + 1] as i64;
        let mut dot = ((w0 * st0 as i64) + (w1 * st1 as i64)) >> 16;
        if dot > 2047 {
            dot = 2047;
        } else if dot < -2047 {
            dot = -2047;
        }
        let mut p = squash(dot as i32);
        if p < 1 {
            p = 1;
        } else if p > 4094 {
            p = 4094;
        }
        // stash transient
        self.tx0 = st0;
        self.tx1 = st1;
        self.wctx = wctx;
        self.mm_idx = mm_idx;
        self.mm_used = mm_used;
        self.pr = p as u32;
        p as u32
    }

    // Update all learned tables with the actual bit.
    #[inline]
    fn update_bit(&mut self, node: u32, i: u32, bit: u32) {
        let target = (bit as i32) << 12; // 0 or 4096
        // order-2 update (identical rule to base)
        let oi = self.idx(node);
        let pr = self.t[oi] as i32;
        self.t[oi] = (pr + ((target - pr) >> 5)) as u16;
        // match-model table update
        if self.mm_used {
            let mp = self.mm[self.mm_idx] as i32;
            self.mm[self.mm_idx] = (mp + ((target - mp) >> 5)) as u16;
        }
        // mixer weight update (logistic; deterministic integer math)
        let err = target - self.pr as i32; // -4095..4095
        let mut nw0 = self.w[self.wctx * 2] + ((self.tx0 * err) >> 12);
        let mut nw1 = self.w[self.wctx * 2 + 1] + ((self.tx1 * err) >> 12);
        if nw0 > WBOUND {
            nw0 = WBOUND;
        } else if nw0 < -WBOUND {
            nw0 = -WBOUND;
        }
        if nw1 > WBOUND {
            nw1 = WBOUND;
        } else if nw1 < -WBOUND {
            nw1 = -WBOUND;
        }
        self.w[self.wctx * 2] = nw0;
        self.w[self.wctx * 2 + 1] = nw1;
        // break the match for the rest of this byte if this bit disagreed
        if self.mm_active {
            let predicted_bit = ((self.match_byte >> i) & 1) as u32;
            if bit != predicted_bit {
                self.mm_active = false;
            }
        }
    }

    // Called once per byte, after all 8 bits are known. Advances context/history/match.
    #[inline]
    fn end_byte(&mut self, b: u8) {
        // order-2 context push
        self.c2 = self.c1;
        self.c1 = b as u32;
        // append to causal history
        self.hist.push(b);
        let len = self.hist.len();
        // advance/break the match run
        if self.mvalid {
            if self.hist[self.mptr] == b {
                self.mlen += 1;
                self.mptr += 1;
                if self.mptr >= len {
                    self.mvalid = false;
                    self.mlen = 0;
                }
            } else {
                self.mvalid = false;
                self.mlen = 0;
            }
        }
        // hash the last 4 bytes; re-acquire a match if we don't have one; then store
        if len >= MINLEN {
            let c0 = self.hist[len - 4] as u32;
            let c1 = self.hist[len - 3] as u32;
            let c2 = self.hist[len - 2] as u32;
            let c3 = self.hist[len - 1] as u32;
            let ctx32 = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3;
            let h = (ctx32.wrapping_mul(2654435761) >> (32 - HTBITS)) as usize;
            if !self.mvalid {
                let cand = self.ht[h];
                if cand != u32::MAX && (cand as usize) < len {
                    self.mptr = cand as usize;
                    self.mvalid = true;
                    self.mlen = 0;
                }
            }
            self.ht[h] = len as u32; // position of the next byte to follow this context
        }
    }
}

// ===================== ARITHMETIC CODER — COPIED VERBATIM FROM BASE =====================

struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}
impl Encoder {
    fn new() -> Self {
        Encoder {
            x1: 0,
            x2: 0xFFFF_FFFF,
            out: Vec::new(),
        }
    }
    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
        // p in [0,4095]; xmid always in [x1, x2-1] since p < 4096.
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p;
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }
    fn flush(&mut self) {
        for _ in 0..4 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    inp: &'a [u8],
    pos: usize,
}
impl<'a> Decoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Decoder {
            x1: 0,
            x2: 0xFFFF_FFFF,
            x: 0,
            inp,
            pos: 0,
        };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next() as u32;
        }
        d
    }
    #[inline]
    fn next(&mut self) -> u8 {
        let b = if self.pos < self.inp.len() {
            self.inp[self.pos]
        } else {
            0
        };
        self.pos += 1;
        b
    }
    #[inline]
    fn decode(&mut self, p: u32) -> u32 {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p;
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.next() as u32;
        }
        bit
    }
}

// ===================== compress / decompress (model wired in) =====================

fn compress(data: &[u8]) -> Vec<u8> {
    let mut m = Model::new();
    let mut e = Encoder::new();
    let n = data.len() as u64;
    let mut archive = Vec::with_capacity(data.len() / 2 + 16);
    archive.extend_from_slice(&n.to_le_bytes()); // 8-byte length header
    for &byte in data {
        m.begin_byte();
        let mut node: u32 = 1;
        for i in (0..8u32).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p = m.predict(node, i);
            e.encode(bit, p);
            m.update_bit(node, i, bit);
            node = (node << 1) | bit;
        }
        m.end_byte(byte);
    }
    e.flush();
    archive.extend_from_slice(&e.out);
    archive
}

fn decompress(archive: &[u8]) -> Vec<u8> {
    let n = u64::from_le_bytes(archive[0..8].try_into().unwrap()) as usize;
    let mut m = Model::new();
    let mut d = Decoder::new(&archive[8..]);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        m.begin_byte();
        let mut node: u32 = 1;
        for i in (0..8u32).rev() {
            let p = m.predict(node, i);
            let bit = d.decode(p);
            m.update_bit(node, i, bit);
            node = (node << 1) | bit;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        m.end_byte(byte);
    }
    out
}

fn decoder_src_bytes() -> u64 {
    // The standalone decompressor cost = this source file's size (honest proxy).
    fs::metadata(file!()).map(|m| m.len()).unwrap_or(0)
}

fn report(mode: &str, input: usize, archive: usize, roundtrip: u32) {
    let dsrc = decoder_src_bytes();
    let total = archive as u64 + dsrc;
    let ratio = if archive > 0 {
        input as f64 / archive as f64
    } else {
        0.0
    };
    let bpc = if input > 0 {
        (archive as f64 * 8.0) / input as f64
    } else {
        0.0
    };
    println!(
        "PILOT|mode={}|input_bytes={}|archive_bytes={}|decoder_src_bytes={}|total_bytes={}|ratio_x={:.4}|bpc={:.4}|roundtrip={}",
        mode, input, archive, dsrc, total, ratio, bpc, roundtrip
    );
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: hp_match <verify|compress|decompress> <in> [out]");
        exit(2);
    }
    match a[1].as_str() {
        "verify" => {
            let data = fs::read(&a[2]).expect("read input");
            let arc = compress(&data);
            let back = decompress(&arc);
            let ok = back == data;
            report("verify", data.len(), arc.len(), ok as u32);
            if !ok {
                eprintln!("ROUNDTRIP_FAIL");
                exit(1);
            }
            println!("ROUNDTRIP_OK");
        }
        "compress" => {
            let data = fs::read(&a[2]).expect("read input");
            let arc = compress(&data);
            fs::write(&a[3], &arc).expect("write archive");
            report("compress", data.len(), arc.len(), 0);
        }
        "decompress" => {
            let arc = fs::read(&a[2]).expect("read archive");
            let out = decompress(&arc);
            fs::write(&a[3], &out).expect("write output");
            println!(
                "DECOMPRESS|archive_bytes={}|output_bytes={}",
                arc.len(),
                out.len()
            );
        }
        _ => {
            eprintln!("unknown mode");
            exit(2);
        }
    }
}
