// hp_dir.rs — key="dir" variant of the Hutter pilot codec.
// LEARNING-CURVE EXPERIMENT: does "more passes / more directions" -> smarter,
// measured on HELD-OUT (second half) only, with a lossless, decoder-replicable protocol.
//
// The binary arithmetic Encoder + Decoder are COPIED VERBATIM from hutter_pilot.rs
// (fpaq0-derived, proven-symmetric). Only the MODEL is replaced: instead of a single
// order-2 context, we mix several independent context PROJECTIONS ("directions") —
// order-1, order-2, order-3, and a strided/skip context — through an integer logistic
// mixer whose weights adapt online. Everything is integer-deterministic so encoder and
// decoder stay byte-exact.
//
// HELD-OUT PROTOCOL (lossless + decoder-replicable):
//   1. Split input: TRAIN = first half, HELDOUT = second half.
//   2. Encode TRAIN normally (single online adaptive pass). The decoder decodes TRAIN
//      first, so BOTH sides then possess TRAIN identically.
//   3. Warmup: both sides perform (passes-1) EXTRA causal sweeps over the now-known
//      TRAIN bytes to strengthen the model. NO bytes are transmitted; fully replicable.
//   4. Encode HELDOUT with the warmed model. heldout_archive_bytes = the ONLY score.
//   The warmup sweeps touch ONLY TRAIN (already known to both sides). compress never
//   lets the model see HELDOUT before encoding it in a way the decoder cannot mirror.
//
// dir-mode controls the SET of directions (the control conditions):
//   expand  = use all distinct directions (order1/order2/order3/skip)  [treatment]
//   dup     = duplicate ONE direction (order2) NDIR times               [repeated-exposure control]
//   shuffle = distinct directions, but per-step the mixer-weight slots are permuted by a
//             deterministic PRNG, scrambling the direction<->weight mapping [bookkeeping control]
//
// Modes:
//   verify     <file> [--passes N] [--dir-mode M] : in-memory FULL-file roundtrip + sizes (asserts exact)
//   compress   <in> <archive> [--passes N] [--dir-mode M] : write archive (header + AC streams)
//   decompress <archive> <out>                    : reconstruct byte-exact (params read from header)
//   curve      <file>                             : run passes in {1,2,4,8,16} x {expand,dup,shuffle},
//                                                   print one CKPT row each (heldout-only score)
//
// PILOT|mode=..|input_bytes=..|archive_bytes=..|decoder_src_bytes=..|total_bytes=..|ratio_x=..|bpc=..|roundtrip=..
// CKPT|passes=..|dir_mode=..|directions=..|heldout_archive_bytes=..|marginal_bits_saved=..|replay_exact=..|rss_kb=..|decode_ms=..|json=0

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Arithmetic coder — COPIED VERBATIM from hutter_pilot.rs (do NOT modify).
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
// End verbatim coder.
// ---------------------------------------------------------------------------

const NDIR: usize = 4;
const TBITS: u32 = 22;
const TSIZE: usize = 1 << TBITS; // 4M u16 = 8MB per direction

#[derive(Clone, Copy, PartialEq, Eq)]
enum DirMode { Expand, Dup, Shuffle }

impl DirMode {
    fn as_u8(self) -> u8 { match self { DirMode::Expand => 0, DirMode::Dup => 1, DirMode::Shuffle => 2 } }
    fn from_u8(v: u8) -> DirMode { match v { 1 => DirMode::Dup, 2 => DirMode::Shuffle, _ => DirMode::Expand } }
    fn name(self) -> &'static str { match self { DirMode::Expand => "expand", DirMode::Dup => "dup", DirMode::Shuffle => "shuffle" } }
    fn parse(s: &str) -> DirMode { match s { "dup" => DirMode::Dup, "shuffle" => DirMode::Shuffle, _ => DirMode::Expand } }
}

// Integer squash: logit d (in ~[-2047,2047]) -> probability in [0,4095].
#[inline]
fn squash(d: i32) -> i32 {
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546,
        2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079,
        4085, 4089, 4092, 4093, 4094,
    ];
    if d > 2047 { return 4095; }
    if d < -2047 { return 0; }
    let w = d & 127;
    let idx = ((d >> 7) + 16) as usize;
    (T[idx] * (128 - w) + T[idx + 1] * w + 64) >> 7
}

// Build the inverse (stretch) table: p in [0,4095] -> logit in [-2047,2047].
fn build_stretch() -> Vec<i32> {
    let mut s = vec![0i32; 4096];
    let mut pi = 0usize;
    for d in -2047..=2047 {
        let p = squash(d) as usize;
        let mut x = pi;
        while x <= p { s[x] = d; x += 1; }
        pi = p + 1;
    }
    for x in pi..4096 { s[x] = 2047; }
    s
}

// Deterministic hash of (slot, context value, node) -> table index.
#[inline]
fn tidx(slot: usize, cv: u32, node: u32) -> usize {
    let mut h = (cv as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (node as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= (slot as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    (h as usize) & (TSIZE - 1)
}

struct Model {
    tabs: Vec<Vec<u16>>,  // NDIR probability tables (12-bit P(bit=1))
    dir_kind: [u8; NDIR], // projection used by each slot: 0=order1 1=order2 2=order3 3=skip
    hist: [u32; 5],       // hist[1]=prev1 .. hist[4]=prev4
    w: Vec<i32>,          // mixer weights, Q16, gated by bit position: [8 * NDIR]
    st_buf: [i32; NDIR],  // stretched inputs for the current bit
    idx_buf: [usize; NDIR],
    wslot: [usize; NDIR], // weight slot chosen for each direction this bit
    bitpos: u32,          // bits emitted in current byte (0..7)
    mode: DirMode,
    lcg: u64,             // PRNG for shuffle mode
    stretch: Vec<i32>,
}

impl Model {
    fn new(mode: DirMode) -> Self {
        let dir_kind = match mode {
            DirMode::Dup => [1u8, 1, 1, 1],  // one direction (order-2) duplicated
            _ => [0u8, 1, 2, 3],             // order1, order2, order3, skip
        };
        let init_w = 65536 / NDIR as i32; // start near an average of stretched inputs
        Model {
            tabs: (0..NDIR).map(|_| vec![2048u16; TSIZE]).collect(),
            dir_kind,
            hist: [0; 5],
            w: vec![init_w; 8 * NDIR],
            st_buf: [0; NDIR],
            idx_buf: [0; NDIR],
            wslot: [0; NDIR],
            bitpos: 0,
            mode,
            lcg: 0x9E37_79B9_7F4A_7C15,
            stretch: build_stretch(),
        }
    }

    #[inline]
    fn reset_hist(&mut self) { self.hist = [0; 5]; }

    #[inline]
    fn push_byte(&mut self, b: u8) {
        self.hist[4] = self.hist[3];
        self.hist[3] = self.hist[2];
        self.hist[2] = self.hist[1];
        self.hist[1] = b as u32;
    }

    #[inline]
    fn context_value(&self, kind: u8) -> u32 {
        match kind {
            0 => self.hist[1],
            1 => self.hist[1] | (self.hist[2] << 8),
            2 => self.hist[1] | (self.hist[2] << 8) | (self.hist[3] << 16),
            _ => self.hist[2] | (self.hist[4] << 8), // skip: positions t-2, t-4
        }
    }

    // Predict P(bit=1) for the current node; records per-direction state for update().
    #[inline]
    fn predict(&mut self, node: u32) -> u32 {
        let base = (self.bitpos as usize) * NDIR;
        let mut perm = [0usize; NDIR];
        for k in 0..NDIR { perm[k] = k; }
        if self.mode == DirMode::Shuffle {
            // Deterministic Fisher-Yates over the weight slots (advances lcg identically
            // on both sides). Scrambles the direction<->weight mapping each bit.
            for k in (1..NDIR).rev() {
                self.lcg = self.lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = ((self.lcg >> 33) as usize) % (k + 1);
                perm.swap(k, j);
            }
        }
        let mut dot: i64 = 0;
        for i in 0..NDIR {
            let cv = self.context_value(self.dir_kind[i]);
            let idx = tidx(i, cv, node);
            self.idx_buf[i] = idx;
            let p = self.tabs[i][idx] as usize;
            let st = self.stretch[p];
            self.st_buf[i] = st;
            let ws = base + perm[i];
            self.wslot[i] = ws;
            dot += (self.w[ws] as i64) * (st as i64);
        }
        let mut d = (dot >> 16) as i32;
        if d > 2047 { d = 2047; }
        if d < -2047 { d = -2047; }
        let mut pr = squash(d);
        if pr < 1 { pr = 1; }
        if pr > 4094 { pr = 4094; }
        pr as u32
    }

    // Update direction tables + mixer weights with the true bit (pr = value from predict()).
    #[inline]
    fn update(&mut self, bit: u32, pr: u32) {
        let err = (((bit as i32) << 12) - pr as i32) * 7;
        let target = (bit as i32) << 12;
        for i in 0..NDIR {
            let idx = self.idx_buf[i];
            let old = self.tabs[i][idx] as i32;
            self.tabs[i][idx] = (old + ((target - old) >> 5)) as u16;
            let ws = self.wslot[i];
            self.w[ws] += (self.st_buf[i] * err + 0x8000) >> 16;
        }
        self.bitpos = (self.bitpos + 1) & 7;
    }
}

// ---------------------------------------------------------------------------
// Byte-loop helpers (shared shape for encode / warmup / decode).
// ---------------------------------------------------------------------------

// Transmit (enc=Some) or warmup-only (enc=None) over known bytes.
fn feed_bytes(m: &mut Model, bytes: &[u8], mut enc: Option<&mut Encoder>) {
    for &byte in bytes {
        let mut node: u32 = 1;
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let pr = m.predict(node);
            if let Some(e) = enc.as_deref_mut() { e.encode(bit, pr); }
            m.update(bit, pr);
            node = (node << 1) | bit;
        }
        m.push_byte(byte);
    }
}

// Decode `count` bytes into dst using decoder d (mirrors feed_bytes transmit path).
fn decode_bytes(m: &mut Model, dst: &mut [u8], d: &mut Decoder) {
    for k in 0..dst.len() {
        let mut node: u32 = 1;
        for _ in 0..8 {
            let pr = m.predict(node);
            let bit = d.decode(pr);
            m.update(bit, pr);
            node = (node << 1) | bit;
        }
        let byte = (node & 0xFF) as u8;
        dst[k] = byte;
        m.push_byte(byte);
    }
}

// ---------------------------------------------------------------------------
// Archive assembly.
//   [0..8]   n  (total bytes)
//   [8..16]  t  (train bytes = n/2)
//   [16..20] passes (u32)
//   [20]     dir_mode (u8)
//   [21..29] train_stream_len (u64)
//   [29 .. 29+tsl]  train AC stream
//   [29+tsl ..]     heldout AC stream
// ---------------------------------------------------------------------------
const HDR: usize = 29;

struct CompRes {
    archive: Vec<u8>,
    heldout_bytes: usize, // = e2 stream length (the held-out score)
}

fn compress(data: &[u8], passes: u32, mode: DirMode) -> CompRes {
    let n = data.len();
    let t = n / 2;
    let train = &data[..t];
    let held = &data[t..];

    let mut m = Model::new(mode);

    // Pass 1: transmit TRAIN online (both sides learn identically while (de)coding).
    let mut e1 = Encoder::new();
    feed_bytes(&mut m, train, Some(&mut e1));
    e1.flush();

    // Warmup: (passes-1) extra causal sweeps over the now-known TRAIN. No bytes emitted.
    for _ in 1..passes {
        m.reset_hist();
        feed_bytes(&mut m, train, None);
    }

    // Encode HELDOUT with the warmed model. History continues from end-of-TRAIN.
    let mut e2 = Encoder::new();
    feed_bytes(&mut m, held, Some(&mut e2));
    e2.flush();

    let tsl = e1.out.len();
    let mut archive = Vec::with_capacity(HDR + e1.out.len() + e2.out.len());
    archive.extend_from_slice(&(n as u64).to_le_bytes());
    archive.extend_from_slice(&(t as u64).to_le_bytes());
    archive.extend_from_slice(&passes.to_le_bytes());
    archive.push(mode.as_u8());
    archive.extend_from_slice(&(tsl as u64).to_le_bytes());
    archive.extend_from_slice(&e1.out);
    archive.extend_from_slice(&e2.out);

    CompRes { archive, heldout_bytes: e2.out.len() }
}

fn decompress(archive: &[u8]) -> Vec<u8> {
    let n = u64::from_le_bytes(archive[0..8].try_into().unwrap()) as usize;
    let t = u64::from_le_bytes(archive[8..16].try_into().unwrap()) as usize;
    let passes = u32::from_le_bytes(archive[16..20].try_into().unwrap());
    let mode = DirMode::from_u8(archive[20]);
    let tsl = u64::from_le_bytes(archive[21..29].try_into().unwrap()) as usize;

    let train_stream = &archive[HDR..HDR + tsl];
    let held_stream = &archive[HDR + tsl..];

    let mut m = Model::new(mode);
    let mut out = vec![0u8; n];

    // Decode TRAIN (mirrors pass 1).
    {
        let mut d1 = Decoder::new(train_stream);
        decode_bytes(&mut m, &mut out[..t], &mut d1);
    }
    // Warmup over the now-known TRAIN (identical to encoder side).
    // out[..t] now equals the original TRAIN.
    let train_known: Vec<u8> = out[..t].to_vec();
    for _ in 1..passes {
        m.reset_hist();
        feed_bytes(&mut m, &train_known, None);
    }
    // Decode HELDOUT with the warmed model.
    {
        let mut d2 = Decoder::new(held_stream);
        decode_bytes(&mut m, &mut out[t..], &mut d2);
    }
    out
}

// ---------------------------------------------------------------------------
// Reporting helpers.
// ---------------------------------------------------------------------------
fn decoder_src_bytes() -> u64 {
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

fn peak_rss_kb() -> u64 {
    if let Ok(s) = fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0);
            }
        }
    }
    0
}

// Parse optional --passes N / --dir-mode M from args.
fn parse_opts(a: &[String]) -> (u32, DirMode) {
    let mut passes = 1u32;
    let mut mode = DirMode::Expand;
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--passes" => { if i + 1 < a.len() { passes = a[i + 1].parse().unwrap_or(1).max(1); i += 1; } }
            "--dir-mode" => { if i + 1 < a.len() { mode = DirMode::parse(&a[i + 1]); i += 1; } }
            _ => {}
        }
        i += 1;
    }
    (passes, mode)
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: hp_dir <verify|compress|decompress|curve> <in> [out] [--passes N] [--dir-mode expand|dup|shuffle]");
        exit(2);
    }
    match a[1].as_str() {
        "verify" => {
            let (passes, mode) = parse_opts(&a[3..]);
            let data = fs::read(&a[2]).expect("read input");
            let cr = compress(&data, passes, mode);
            let back = decompress(&cr.archive);
            let ok = back == data;
            report("verify", data.len(), cr.archive.len(), ok as u32);
            println!(
                "HELDOUT|passes={}|dir_mode={}|directions={}|heldout_archive_bytes={}|full_archive_bytes={}|roundtrip={}|json=0",
                passes, mode.name(), NDIR, cr.heldout_bytes, cr.archive.len(), ok as u32
            );
            if !ok { eprintln!("ROUNDTRIP_FAIL"); exit(1); }
            println!("ROUNDTRIP_OK");
        }
        "compress" => {
            let (passes, mode) = parse_opts(&a[4.min(a.len())..]);
            let data = fs::read(&a[2]).expect("read input");
            let cr = compress(&data, passes, mode);
            fs::write(&a[3], &cr.archive).expect("write archive");
            report("compress", data.len(), cr.archive.len(), 0);
            println!(
                "HELDOUT|passes={}|dir_mode={}|directions={}|heldout_archive_bytes={}|json=0",
                passes, mode.name(), NDIR, cr.heldout_bytes
            );
        }
        "decompress" => {
            let arc = fs::read(&a[2]).expect("read archive");
            let out = decompress(&arc);
            fs::write(&a[3], &out).expect("write output");
            println!("DECOMPRESS|archive_bytes={}|output_bytes={}", arc.len(), out.len());
        }
        "curve" => {
            let data = fs::read(&a[2]).expect("read input");
            let pass_list = [1u32, 2, 4, 8, 16];
            for mode in [DirMode::Expand, DirMode::Dup, DirMode::Shuffle] {
                let mut prev: Option<usize> = None;
                for &passes in pass_list.iter() {
                    let cr = compress(&data, passes, mode);
                    let t0 = Instant::now();
                    let back = decompress(&cr.archive);
                    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let replay = (back == data) as u32;
                    let ho = cr.heldout_bytes;
                    let marginal = match prev {
                        Some(p) => (p as i64 - ho as i64) * 8,
                        None => 0,
                    };
                    prev = Some(ho);
                    println!(
                        "CKPT|passes={}|dir_mode={}|directions={}|heldout_archive_bytes={}|marginal_bits_saved={}|replay_exact={}|rss_kb={}|decode_ms={:.1}|json=0",
                        passes, mode.name(), NDIR, ho, marginal, replay, peak_rss_kb(), decode_ms
                    );
                }
            }
        }
        _ => { eprintln!("unknown mode"); exit(2); }
    }
}
