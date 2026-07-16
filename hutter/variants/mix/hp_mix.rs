// hp_mix.rs — dep-free lossless codec, Hutter pre-test variant key="mix".
// Binary arithmetic coder (fpaq0-derived, proven-symmetric) copied VERBATIM
// from hutter_pilot.rs. Only the MODEL is replaced: logistic context mixing
// (lpaq-lite) over order-0..4 byte contexts + a bias input, mixed with an
// integer (16.16 fixed-point) neural mixer whose weights are selected by the
// previous byte. Fully deterministic (no RNG, no f64): compress and decompress
// run the IDENTICAL predict+update path so probabilities match bit-for-bit.
//
// Modes:
//   verify     <file>            : in-memory roundtrip + print sizes/bpc (asserts exact)
//   compress   <in> <archive>    : write archive (8-byte LE length header + AC stream)
//   decompress <archive> <out>   : reconstruct byte-exact
//
// Reports one machine row:  PILOT|mode=..|input_bytes=..|archive_bytes=..|
//                           decoder_src_bytes=..|total_bytes=..|ratio_x=..|bpc=..|roundtrip=..

use std::env;
use std::fs;
use std::process::exit;

// ---------------------------------------------------------------------------
// squash / stretch (logistic <-> 12-bit probability), lpaq tables.
// ---------------------------------------------------------------------------
#[inline]
fn squash(d: i32) -> i32 {
    // maps a stretched value in [-2047,2047] to a 12-bit probability 0..4095.
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047,
        2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085,
        4089, 4092, 4093, 4094,
    ];
    if d > 2047 { return 4095; }
    if d < -2047 { return 0; }
    let w = d & 127;
    let idx = ((d >> 7) + 16) as usize;
    (T[idx] * (128 - w) + T[idx + 1] * w + 64) >> 7
}

fn build_stretch() -> Vec<i32> {
    // stretch[p] = inverse of squash, precomputed over p in 0..4095.
    let mut stretch = vec![0i32; 4096];
    let mut pi: i32 = 0;
    for x in -2047..=2047 {
        let p = squash(x);
        while pi <= p {
            stretch[pi as usize] = x;
            pi += 1;
        }
    }
    while (pi as usize) < 4096 {
        stretch[pi as usize] = 2047;
        pi += 1;
    }
    stretch
}

// ---------------------------------------------------------------------------
// MODEL: logistic context mixing over order-0..4 byte contexts + bias.
// ---------------------------------------------------------------------------
const NIN: usize = 6; // 5 context inputs (orders 0..4) + 1 bias input
const O0SIZE: usize = 256; // order-0: node only
const O1SIZE: usize = 1 << 16; // order-1: (c1<<8)|node, exact
const HSIZE: usize = 1 << 22; // hashed tables for orders 2..4
const MC: usize = 256; // mixer weight contexts (selected by prev byte)
const RATE: i32 = 4; // counter adaptation rate (>> shift)
const LR_SHIFT: i32 = 12; // mixer learning-rate (>> shift)
const WCLAMP: i32 = 1 << 24; // weight clamp to guarantee no i32 overflow

struct Model {
    t0: Vec<u16>,
    t1: Vec<u16>,
    t2: Vec<u16>,
    t3: Vec<u16>,
    t4: Vec<u16>,
    w: Vec<i32>, // MC * NIN weights, 16.16 fixed point
    // byte history
    c1: u32,
    c2: u32,
    c3: u32,
    c4: u32,
    // per-byte base hashes for orders 2..4
    h2: u32,
    h3: u32,
    h4: u32,
    // per-bit scratch (set in predict, consumed in update)
    idx: [usize; 5],
    st: [i32; NIN],
    wctx: usize,
    pr: i32,
    stretch: Vec<i32>,
}

#[inline]
fn hidx(h: u32, node: u32, size: usize) -> usize {
    let x = (h ^ node.wrapping_mul(0x9E37_79B1)).wrapping_mul(0x2545_F491);
    (x as usize) & (size - 1)
}

impl Model {
    fn new() -> Self {
        let mut w = vec![0i32; MC * NIN];
        // init the 5 context-input weights so the initial prediction is the
        // average of the inputs; leave the bias weight at 0.
        let init = (65536 / 5) as i32;
        for c in 0..MC {
            for i in 0..5 {
                w[c * NIN + i] = init;
            }
        }
        Model {
            t0: vec![2048u16; O0SIZE],
            t1: vec![2048u16; O1SIZE],
            t2: vec![2048u16; HSIZE],
            t3: vec![2048u16; HSIZE],
            t4: vec![2048u16; HSIZE],
            w,
            c1: 0,
            c2: 0,
            c3: 0,
            c4: 0,
            h2: 0,
            h3: 0,
            h4: 0,
            idx: [0; 5],
            st: [0; NIN],
            wctx: 0,
            pr: 2048,
            stretch: build_stretch(),
        }
    }

    #[inline]
    fn predict(&mut self, node: u32) -> u32 {
        self.idx[0] = (node as usize) & (O0SIZE - 1);
        self.idx[1] = (((self.c1 << 8) | node) as usize) & (O1SIZE - 1);
        self.idx[2] = hidx(self.h2, node, HSIZE);
        self.idx[3] = hidx(self.h3, node, HSIZE);
        self.idx[4] = hidx(self.h4, node, HSIZE);

        let p0 = self.t0[self.idx[0]] as usize;
        let p1 = self.t1[self.idx[1]] as usize;
        let p2 = self.t2[self.idx[2]] as usize;
        let p3 = self.t3[self.idx[3]] as usize;
        let p4 = self.t4[self.idx[4]] as usize;

        self.st[0] = self.stretch[p0];
        self.st[1] = self.stretch[p1];
        self.st[2] = self.stretch[p2];
        self.st[3] = self.stretch[p3];
        self.st[4] = self.stretch[p4];
        self.st[5] = 256; // bias input (constant)

        self.wctx = (self.c1 as usize) & (MC - 1);
        let base = self.wctx * NIN;
        let mut dot: i64 = 0;
        for i in 0..NIN {
            dot += (self.st[i] as i64) * (self.w[base + i] as i64);
        }
        let mut d = (dot >> 16) as i32;
        if d > 2047 { d = 2047; }
        if d < -2047 { d = -2047; }
        let mut p = squash(d);
        if p < 1 { p = 1; }
        if p > 4094 { p = 4094; }
        self.pr = p;
        p as u32
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let target = (bit as i32) << 12;
        // counter updates (move each context prob toward the observed bit)
        {
            let t = &mut self.t0[self.idx[0]];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        {
            let t = &mut self.t1[self.idx[1]];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        {
            let t = &mut self.t2[self.idx[2]];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        {
            let t = &mut self.t3[self.idx[3]];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        {
            let t = &mut self.t4[self.idx[4]];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        // mixer weight update: w += (input * err) >> LR_SHIFT
        let err = target - self.pr; // -4093..4093
        let base = self.wctx * NIN;
        for i in 0..NIN {
            let mut nw = self.w[base + i] + ((self.st[i] * err) >> LR_SHIFT);
            if nw > WCLAMP { nw = WCLAMP; }
            if nw < -WCLAMP { nw = -WCLAMP; }
            self.w[base + i] = nw;
        }
    }

    #[inline]
    fn push_byte(&mut self, b: u8) {
        self.c4 = self.c3;
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = b as u32;
        self.h2 = self
            .c1
            .wrapping_mul(0x6B43_A9B5)
            .wrapping_add(self.c2.wrapping_add(1).wrapping_mul(0x9E37_79B1));
        self.h3 = self
            .h2
            .wrapping_mul(0x2545_F491)
            .wrapping_add(self.c3.wrapping_add(1).wrapping_mul(0x85EB_CA77));
        self.h4 = self
            .h3
            .wrapping_mul(0x2545_F491)
            .wrapping_add(self.c4.wrapping_add(1).wrapping_mul(0xC2B2_AE35));
    }
}

// ---------------------------------------------------------------------------
// Arithmetic coder — COPIED VERBATIM from hutter_pilot.rs (do not modify).
// ---------------------------------------------------------------------------
struct Encoder { x1: u32, x2: u32, out: Vec<u8> }
impl Encoder {
    fn new() -> Self { Encoder { x1: 0, x2: 0xFFFF_FFFF, out: Vec::new() } }
    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
        // p in [0,4095]; xmid always in [x1, x2-1] since p < 4096.
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p;
        if bit == 1 { self.x2 = xmid; } else { self.x1 = xmid + 1; }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }
    fn flush(&mut self) {
        for _ in 0..4 { self.out.push((self.x1 >> 24) as u8); self.x1 <<= 8; }
    }
}

struct Decoder<'a> { x1: u32, x2: u32, x: u32, inp: &'a [u8], pos: usize }
impl<'a> Decoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: 0xFFFF_FFFF, x: 0, inp, pos: 0 };
        for _ in 0..4 { d.x = (d.x << 8) | d.next() as u32; }
        d
    }
    #[inline]
    fn next(&mut self) -> u8 {
        let b = if self.pos < self.inp.len() { self.inp[self.pos] } else { 0 };
        self.pos += 1; b
    }
    #[inline]
    fn decode(&mut self, p: u32) -> u32 {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p;
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit == 1 { self.x2 = xmid; } else { self.x1 = xmid + 1; }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.next() as u32;
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// compress / decompress — same structure/CLI as the base pilot.
// ---------------------------------------------------------------------------
fn compress(data: &[u8]) -> Vec<u8> {
    let mut m = Model::new();
    let mut e = Encoder::new();
    let n = data.len() as u64;
    let mut archive = Vec::with_capacity(data.len() / 2 + 16);
    archive.extend_from_slice(&n.to_le_bytes()); // 8-byte length header
    for &byte in data {
        let mut node: u32 = 1;
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p = m.predict(node);
            e.encode(bit, p);
            m.update(bit);
            node = (node << 1) | bit;
        }
        m.push_byte(byte);
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
        let mut node: u32 = 1;
        for _ in 0..8 {
            let p = m.predict(node);
            let bit = d.decode(p);
            m.update(bit);
            node = (node << 1) | bit;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        m.push_byte(byte);
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
    let ratio = if archive > 0 { input as f64 / archive as f64 } else { 0.0 };
    let bpc = if input > 0 { (archive as f64 * 8.0) / input as f64 } else { 0.0 };
    println!(
        "PILOT|mode={}|input_bytes={}|archive_bytes={}|decoder_src_bytes={}|total_bytes={}|ratio_x={:.4}|bpc={:.4}|roundtrip={}",
        mode, input, archive, dsrc, total, ratio, bpc, roundtrip
    );
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 { eprintln!("usage: hp_mix <verify|compress|decompress> <in> [out]"); exit(2); }
    match a[1].as_str() {
        "verify" => {
            let data = fs::read(&a[2]).expect("read input");
            let arc = compress(&data);
            let back = decompress(&arc);
            let ok = back == data;
            report("verify", data.len(), arc.len(), ok as u32);
            if !ok { eprintln!("ROUNDTRIP_FAIL"); exit(1); }
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
            println!("DECOMPRESS|archive_bytes={}|output_bytes={}", arc.len(), out.len());
        }
        _ => { eprintln!("unknown mode"); exit(2); }
    }
}
