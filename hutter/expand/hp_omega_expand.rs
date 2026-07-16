// hutter_omega_expand.rs — v2 whole-Monty DIRECTIONAL-EXPANSION codec (impl A).
//
// A context-mixing arithmetic coder whose many context inputs ARE the Asolaria
// directions ("omega cubes training together, separately"). We code RAW BYTES
// (structure-preserving). Each direction is a separate adaptive bit-predictor
// derived from ALREADY-CODED history; the logistic mixer fuses them IN PARALLEL.
//
// - fpaq0 Encoder/Decoder copied VERBATIM from hp_mix.rs (byte-identical).
// - stretch/squash tables + integer logistic mixer + order-0..4 byte contexts
//   copied from hp_mix.rs as the BASE inputs.
// - A-view transforms (revb/g_n/rol/blkrev/evo/qprism-perm) adapted from
//   unified_omega.rs and applied to CAUSAL HISTORY to DERIVE CONTEXTS — the
//   coded stream is NEVER permuted/repacked (that killed v1).
//
// Directions are PARALLEL context features, not sequential warmup retraining.
//
// Modes:
//   verify     <in> [--dirs K] [--dir-mode M]  : in-memory roundtrip + PILOT row
//   compress   <in> <archive> [--dirs K] [--dir-mode M]
//   decompress <archive> <out>                 : params read from header
//   curve      <in> [--passes-max]             : expansion sweep, CKPT rows
//   selftest                                   : edge cases (empty/1B/random/repeated)
//
// Score = archive_bytes + decoder_src_bytes; archive_ratio=NOT_CLAIMED; claims_final_apex=0.

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

// ===========================================================================
// sha256 (from unified_omega.rs) — seeds the ROOT + direction constants.
// ===========================================================================
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut msg = data.to_vec();
    let bl = (msg.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bl.to_be_bytes());
    let mut h: [u32;8]=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for ch in msg.chunks_exact(64) {
        let mut w=[0u32;64];
        for i in 0..16 { w[i]=u32::from_be_bytes(ch[i*4..i*4+4].try_into().unwrap()); }
        for i in 16..64 {
            let s0=w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);
            let s1=w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh)=(h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let cbh=(e&f)^((!e)&g);
            let t1=hh.wrapping_add(s1).wrapping_add(cbh).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c);
            let t2=s0.wrapping_add(maj);
            hh=g;g=f;f=e;e=d.wrapping_add(t1);d=c;c=b;b=a;a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a);h[1]=h[1].wrapping_add(b);h[2]=h[2].wrapping_add(c);h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e);h[5]=h[5].wrapping_add(f);h[6]=h[6].wrapping_add(g);h[7]=h[7].wrapping_add(hh);
    }
    let mut o=[0u8;32];
    for (i,v) in h.iter().enumerate(){o[i*4..i*4+4].copy_from_slice(&v.to_be_bytes());}
    o
}
fn hex(b:&[u8])->String{const H:&[u8;16]=b"0123456789abcdef";let mut s=String::with_capacity(b.len()*2);for &x in b{s.push(H[(x>>4)as usize]as char);s.push(H[(x&15)as usize]as char);}s}

// ===========================================================================
// ROOT + direction seed constants (charged nothing; decoder regenerates).
// ===========================================================================
fn root_hash() -> [u8;32] { sha256(b"ASOLARIA-OMEGA-EXPAND-V2|8467a937cba309f7") }
fn root_seeds() -> Vec<u32> {
    let mut seeds = Vec::with_capacity(64);
    let mut cur = root_hash();
    for _ in 0..8 {
        for i in 0..8 { seeds.push(u32::from_le_bytes(cur[i*4..i*4+4].try_into().unwrap())); }
        cur = sha256(&cur);
    }
    seeds
}

// Omega ladder commitment (floors 6/8/10/12) folded to UNIFIEDOMEGA — bookkeeping,
// regenerated from ROOT, charged nothing, does not touch the archive.
fn omega_ladder() -> (Vec<(u32,String)>, String) {
    let root = root_hash();
    let mut floor_omegas = Vec::new();
    let mut hexes: Vec<String> = Vec::new();
    for &bits in &[6u32,8,10,12] {
        let mut m = b"OMEGA-EXPAND-FLOOR|".to_vec();
        m.extend_from_slice(&root);
        m.extend_from_slice(&bits.to_le_bytes());
        let h = hex(&sha256(&m));
        floor_omegas.push((bits, h.clone()));
        hexes.push(h);
    }
    hexes.sort();
    let mut m = b"OMEGA-EXPAND-UNIFIED\0".to_vec();
    m.extend_from_slice(&root);
    for h in &hexes { m.extend_from_slice(h.as_bytes()); }
    let unified = hex(&sha256(&m));
    (floor_omegas, unified)
}

// ===========================================================================
// squash / stretch — VERBATIM from hp_mix.rs.
// ===========================================================================
#[inline]
fn squash(d: i32) -> i32 {
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
    let mut stretch = vec![0i32; 4096];
    let mut pi: i32 = 0;
    for x in -2047..=2047 {
        let p = squash(x);
        while pi <= p { stretch[pi as usize] = x; pi += 1; }
    }
    while (pi as usize) < 4096 { stretch[pi as usize] = 2047; pi += 1; }
    stretch
}

// ===========================================================================
// Direction context features (derived from CAUSAL history bytes only).
// win[0] = most recent byte (c1), win[1] = c2, ... up to WLEN back.
// ===========================================================================
const WLEN: usize = 8;
const NDIR: usize = 40;

#[inline] fn revbyte(b: u8) -> u8 { let mut r=0u8; for i in 0..8 { if b&(1<<i)!=0 { r|=1<<(7-i); } } r }
#[inline] fn rol8(b: u8) -> u8 { (b<<1)|(b>>7) }
#[inline] fn swapnib(b: u8) -> u8 { (b<<4)|(b>>4) }

#[inline]
fn mixhash(seed: u32, vals: &[u32]) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    for &v in vals {
        h ^= v.wrapping_add(0x9E37_79B1);
        h = h.wrapping_mul(0x2545_F491);
        h ^= h >> 15;
    }
    h.wrapping_mul(0x85EB_CA77)
}

fn perm_of(n: usize, seed: u32) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = ((s >> 8) as usize) % (i + 1);
        p.swap(i, j);
    }
    p
}

// 12 TRI sector offset-sets (1-indexed distances back).
const SECTORS: [&[usize]; 12] = [
    &[1,2], &[1,3], &[1,4], &[2,3], &[2,4], &[3,4],
    &[1,5], &[1,6], &[2,5], &[1,2,3], &[1,2,4], &[2,4,6],
];
// PI lens grid: 4 floor-widths x 5 window byte-counts = 20 lenses.
const LENS_BITS: [u32; 4] = [6, 8, 10, 12];
const LENS_NB: [usize; 5] = [2, 3, 4, 6, 8];

fn lens_vals(win: &[u8; WLEN], nb: usize, b: u32) -> Vec<u32> {
    let mut bits: u64 = 0;
    for k in 0..nb { bits = (bits << 8) | (win[k] as u64); }
    let totbits = (nb * 8) as u32;
    let groups = totbits / b;
    let mask = (1u64 << b) - 1;
    let mut out = Vec::with_capacity(groups as usize);
    for g in 0..groups {
        let shift = totbits - (g + 1) * b;
        out.push(((bits >> shift) & mask) as u32);
    }
    out
}

// Compute all 40 canonical direction hashes from the causal window.
fn all_dir_hashes(win: &[u8; WLEN], seeds: &[u32], perm8: &[usize]) -> [u32; NDIR] {
    let mut dh = [0u32; NDIR];
    let w = |k: usize| win[k] as u32;
    // CUBE_8POLE (0..8): 8 A-view projections of causal history.
    dh[0] = mixhash(seeds[0], &[w(0),w(1),w(2),w(3),w(4),w(5)]);                       // id / order-6
    dh[1] = mixhash(seeds[1], &[revbyte(win[0])as u32,revbyte(win[1])as u32,revbyte(win[2])as u32,revbyte(win[3])as u32,revbyte(win[4])as u32]); // revb
    dh[2] = mixhash(seeds[2], &[(win[0]^win[1])as u32,(win[1]^win[2])as u32,(win[2]^win[3])as u32,(win[3]^win[4])as u32,(win[4]^win[5])as u32]); // xor-delta
    dh[3] = mixhash(seeds[3], &[rol8(win[0])as u32,rol8(win[1])as u32,rol8(win[2])as u32,rol8(win[3])as u32,rol8(win[4])as u32]);                // rol
    dh[4] = mixhash(seeds[4], &[swapnib(win[0])as u32,swapnib(win[1])as u32,swapnib(win[2])as u32,swapnib(win[3])as u32,swapnib(win[4])as u32]); // g_n half-swap
    dh[5] = mixhash(seeds[5], &[w(7),w(6),w(5),w(4),w(3),w(2),w(1),w(0)]);              // blkrev (reversed window)
    dh[6] = mixhash(seeds[6], &[w(0),w(2),w(4),w(6)]);                                  // evo even-stride
    { let mut v = [0u32; WLEN]; for k in 0..WLEN { v[k] = win[perm8[k]] as u32; }
      dh[7] = mixhash(seeds[7], &v); }                                                  // qprism perm
    // TRI_12SECTOR (8..20): skip/stride sector contexts.
    for (j, set) in SECTORS.iter().enumerate() {
        let mut v: Vec<u32> = Vec::with_capacity(set.len());
        for &off in set.iter() { v.push(win[off - 1] as u32); }
        dh[8 + j] = mixhash(seeds[8 + j], &v);
    }
    // PI_20LENS (20..40): multi-scale floor lenses.
    for (bi, &b) in LENS_BITS.iter().enumerate() {
        for (ni, &nb) in LENS_NB.iter().enumerate() {
            let idx = 20 + bi * 5 + ni;
            let v = lens_vals(win, nb, b);
            dh[idx] = mixhash(seeds[idx], &v);
        }
    }
    dh
}

// ===========================================================================
// MODEL: order-0..4 base inputs + K direction inputs + bias, logistic mixer.
// ===========================================================================
const NIN: usize = 5 + NDIR + 1; // 5 base orders + 40 directions + bias  = 46
const BIAS_IX: usize = NIN - 1;  // 45
const O0SIZE: usize = 256;
const O1SIZE: usize = 1 << 16;
const HSIZE: usize = 1 << 22;    // base orders 2..4
const DSIZE: usize = 1 << 21;    // per-direction table
const MC: usize = 256;
const RATE: i32 = 4;
const LR_SHIFT: i32 = 12;
const WCLAMP: i32 = 1 << 24;

#[derive(Clone, Copy, PartialEq)]
enum DirMode { Expand, Dup, Shuffle }
fn dirmode_code(m: DirMode) -> u8 { match m { DirMode::Expand=>0, DirMode::Dup=>1, DirMode::Shuffle=>2 } }
fn dirmode_from(c: u8) -> DirMode { match c { 1=>DirMode::Dup, 2=>DirMode::Shuffle, _=>DirMode::Expand } }
fn dirmode_name(m: DirMode) -> &'static str { match m { DirMode::Expand=>"expand", DirMode::Dup=>"dup", DirMode::Shuffle=>"shuffle" } }

struct Model {
    t0: Vec<u16>, t1: Vec<u16>, t2: Vec<u16>, t3: Vec<u16>, t4: Vec<u16>,
    dtab: Vec<Vec<u16>>,       // K direction tables
    w: Vec<i32>,               // MC * NIN weights, 16.16 fixed point
    // byte history / base order hashes
    c1: u32, c2: u32, c3: u32, c4: u32,
    h2: u32, h3: u32, h4: u32,
    // direction machinery
    dirs: usize,
    mode: DirMode,
    seeds: Vec<u32>,
    perm8: Vec<usize>,
    slot_src: Vec<usize>,      // slot -> source direction index (len dirs)
    win: [u8; WLEN],
    eff: [u32; NDIR],          // per-byte effective base hash per slot
    // per-bit scratch
    idx: [usize; 5],
    didx: [usize; NDIR],
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
    fn new(dirs: usize, mode: DirMode) -> Self {
        let dirs = dirs.min(NDIR);
        let seeds = root_seeds();
        let perm8 = perm_of(WLEN, seeds[10]);
        // slot -> source direction mapping per dir-mode.
        let slot_src: Vec<usize> = match mode {
            DirMode::Expand  => (0..dirs).collect(),
            DirMode::Dup     => vec![0usize; dirs],
            DirMode::Shuffle => {
                let p = perm_of(NDIR, seeds[11] ^ (dirs as u32));
                p.into_iter().take(dirs).collect()
            }
        };
        let mut w = vec![0i32; MC * NIN];
        let init = if dirs + 5 > 0 { (65536 / (5 + dirs)) as i32 } else { 0 };
        for c in 0..MC {
            for i in 0..5 { w[c * NIN + i] = init; }
            for i in 5..5 + dirs { w[c * NIN + i] = init; }
        }
        let dtab = (0..dirs).map(|_| vec![2048u16; DSIZE]).collect();
        Model {
            t0: vec![2048u16; O0SIZE], t1: vec![2048u16; O1SIZE],
            t2: vec![2048u16; HSIZE], t3: vec![2048u16; HSIZE], t4: vec![2048u16; HSIZE],
            dtab, w,
            c1:0,c2:0,c3:0,c4:0,h2:0,h3:0,h4:0,
            dirs, mode, seeds, perm8, slot_src,
            win: [0u8; WLEN], eff: [0u32; NDIR],
            idx: [0;5], didx: [0;NDIR], st: [0;NIN], wctx: 0, pr: 2048,
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

        self.st[0] = self.stretch[self.t0[self.idx[0]] as usize];
        self.st[1] = self.stretch[self.t1[self.idx[1]] as usize];
        self.st[2] = self.stretch[self.t2[self.idx[2]] as usize];
        self.st[3] = self.stretch[self.t3[self.idx[3]] as usize];
        self.st[4] = self.stretch[self.t4[self.idx[4]] as usize];
        // direction inputs (parallel)
        for s in 0..self.dirs {
            let di = hidx(self.eff[s], node, DSIZE);
            self.didx[s] = di;
            self.st[5 + s] = self.stretch[self.dtab[s][di] as usize];
        }
        // inactive direction slots stay 0
        for s in self.dirs..NDIR { self.st[5 + s] = 0; }
        self.st[BIAS_IX] = 256;

        self.wctx = (self.c1 as usize) & (MC - 1);
        let base = self.wctx * NIN;
        let mut dot: i64 = 0;
        for i in 0..NIN { dot += (self.st[i] as i64) * (self.w[base + i] as i64); }
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
        macro_rules! upd { ($t:expr, $ix:expr) => {{ let t=&mut $t[$ix]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }} }
        upd!(self.t0, self.idx[0]);
        upd!(self.t1, self.idx[1]);
        upd!(self.t2, self.idx[2]);
        upd!(self.t3, self.idx[3]);
        upd!(self.t4, self.idx[4]);
        for s in 0..self.dirs {
            let di = self.didx[s];
            let t = &mut self.dtab[s][di];
            let pr = *t as i32;
            *t = (pr + ((target - pr) >> RATE)) as u16;
        }
        let err = target - self.pr;
        let base = self.wctx * NIN;
        for i in 0..NIN {
            if self.st[i] == 0 { continue; } // inactive dirs (and any zero input) => no change
            let mut nw = self.w[base + i] + ((self.st[i] * err) >> LR_SHIFT);
            if nw > WCLAMP { nw = WCLAMP; }
            if nw < -WCLAMP { nw = -WCLAMP; }
            self.w[base + i] = nw;
        }
    }

    #[inline]
    fn push_byte(&mut self, b: u8) {
        self.c4 = self.c3; self.c3 = self.c2; self.c2 = self.c1; self.c1 = b as u32;
        self.h2 = self.c1.wrapping_mul(0x6B43_A9B5)
            .wrapping_add(self.c2.wrapping_add(1).wrapping_mul(0x9E37_79B1));
        self.h3 = self.h2.wrapping_mul(0x2545_F491)
            .wrapping_add(self.c3.wrapping_add(1).wrapping_mul(0x85EB_CA77));
        self.h4 = self.h3.wrapping_mul(0x2545_F491)
            .wrapping_add(self.c4.wrapping_add(1).wrapping_mul(0xC2B2_AE35));
        // shift the causal window (win[0] = newest)
        for k in (1..WLEN).rev() { self.win[k] = self.win[k-1]; }
        self.win[0] = b;
        // recompute the 40 canonical direction hashes, then map to active slots
        let dh = all_dir_hashes(&self.win, &self.seeds, &self.perm8);
        for s in 0..self.dirs {
            let src = self.slot_src[s];
            self.eff[s] = match self.mode {
                DirMode::Dup => dh[0] ^ (s as u32).wrapping_mul(0x9E37_79B1),
                _ => dh[src],
            };
        }
    }

    // Adapt over a byte slice WITHOUT emitting (used only by curve warm/train).
    fn train_pass(&mut self, data: &[u8]) {
        for &byte in data {
            let mut node: u32 = 1;
            for i in (0..8).rev() {
                let bit = ((byte >> i) & 1) as u32;
                let _ = self.predict(node);
                self.update(bit);
                node = (node << 1) | bit;
            }
            self.push_byte(byte);
        }
    }
}

// ===========================================================================
// Arithmetic coder — COPIED VERBATIM from hp_mix.rs (do not modify).
// ===========================================================================
struct Encoder { x1: u32, x2: u32, out: Vec<u8> }
impl Encoder {
    fn new() -> Self { Encoder { x1: 0, x2: 0xFFFF_FFFF, out: Vec::new() } }
    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
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

// ===========================================================================
// compress / decompress.  Header: 8B LE len | dirs u16 LE | dirmode u8 | passes u8
// ===========================================================================
fn compress(data: &[u8], dirs: usize, mode: DirMode) -> Vec<u8> {
    let mut m = Model::new(dirs, mode);
    let mut e = Encoder::new();
    let n = data.len() as u64;
    let mut archive = Vec::with_capacity(data.len() / 2 + 16);
    archive.extend_from_slice(&n.to_le_bytes());
    archive.extend_from_slice(&(dirs.min(NDIR) as u16).to_le_bytes());
    archive.push(dirmode_code(mode));
    archive.push(1u8); // passes (real compression is a single causal pass)
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
    let dirs = u16::from_le_bytes(archive[8..10].try_into().unwrap()) as usize;
    let mode = dirmode_from(archive[10]);
    // archive[11] = passes (unused in causal decode)
    let mut m = Model::new(dirs, mode);
    let mut d = Decoder::new(&archive[12..]);
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

// Encode a slice with an ALREADY-WARMED model (continues adapting). For curve.
fn encode_with(m: &mut Model, data: &[u8]) -> Vec<u8> {
    let mut e = Encoder::new();
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
    e.out
}
fn decode_with(m: &mut Model, stream: &[u8], n: usize) -> Vec<u8> {
    let mut d = Decoder::new(stream);
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

// ===========================================================================
// reporting helpers.
// ===========================================================================
fn decoder_src_bytes() -> u64 {
    fs::metadata(file!()).map(|m| m.len()).unwrap_or(0)
}
fn rss_kb() -> u64 {
    if let Ok(s) = fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                for tok in rest.split_whitespace() {
                    if let Ok(v) = tok.parse::<u64>() { return v; }
                }
            }
        }
    }
    0
}
fn print_omega() {
    let (floors, unified) = omega_ladder();
    for (bits, h) in &floors {
        println!("FLOOROMEGA|floor_bits={}|floor={}|floor_omega={}|json=0", bits, 1u32<<bits, h);
    }
    println!("UNIFIEDOMEGA|method=H(root||sorted[floor6,floor8,floor10,floor12])|omega_unified={}|charged=0|json=0", unified);
}
fn report(mode: &str, input: usize, archive: usize, roundtrip: u32, dirs: usize, dmode: DirMode) {
    let dsrc = decoder_src_bytes();
    let total = archive as u64 + dsrc;
    let ratio = if archive > 0 { input as f64 / archive as f64 } else { 0.0 };
    let bpc = if input > 0 { (archive as f64 * 8.0) / input as f64 } else { 0.0 };
    println!(
        "PILOT|mode={}|dirs={}|dir_mode={}|input_bytes={}|archive_bytes={}|decoder_src_bytes={}|total_bytes={}|ratio_x={:.4}|bpc={:.4}|archive_ratio=NOT_CLAIMED|claims_final_apex=0|roundtrip={}",
        mode, dirs, dirmode_name(dmode), input, archive, dsrc, total, ratio, bpc, roundtrip
    );
}

// ===========================================================================
// curve — expansion sweep over dirs x passes x dir-mode. TRAIN=1st half, HELDOUT=2nd.
// ===========================================================================
fn curve(data: &[u8]) {
    let half = data.len() / 2;
    let train = &data[..half];
    let held = &data[half..];
    println!("CURVE_SETUP|total_bytes={}|train_bytes={}|heldout_bytes={}|json=0", data.len(), train.len(), held.len());
    print_omega();
    let dirs_axis = [4usize, 8, 20, 40];
    let passes_axis = [1usize, 2, 4];
    let modes = [DirMode::Expand, DirMode::Dup, DirMode::Shuffle];
    for &mode in &modes {
        for &passes in &passes_axis {
            let mut prev: Option<usize> = None;
            for &dirs in &dirs_axis {
                let t0 = Instant::now();
                // warm model over TRAIN `passes` times, then encode HELDOUT.
                let mut enc = Model::new(dirs, mode);
                for _ in 0..passes { enc.train_pass(train); }
                let stream = encode_with(&mut enc, held);
                let encode_ms = t0.elapsed().as_millis();
                let arc = stream.len();
                // replay: identically-warmed decoder must restore HELDOUT exactly.
                let mut dec = Model::new(dirs, mode);
                for _ in 0..passes { dec.train_pass(train); }
                let back = decode_with(&mut dec, &stream, held.len());
                let replay_exact = (back == held) as u32;
                let marginal = match prev { Some(p) => arc as i64 - p as i64, None => 0 };
                println!(
                    "CKPT|dirs={}|passes={}|dir_mode={}|heldout_archive_bytes={}|marginal_vs_prev_dirs={}|replay_exact={}|rss_kb={}|encode_ms={}|json=0",
                    dirs, passes, dirmode_name(mode), arc, marginal, replay_exact, rss_kb(), encode_ms
                );
                prev = Some(arc);
            }
        }
    }
}

// ===========================================================================
// selftest — edge cases (empty / 1-byte / random / repeated) at dirs 8 and 40.
// ===========================================================================
fn rt_ok(data: &[u8], dirs: usize, mode: DirMode) -> bool {
    let arc = compress(data, dirs, mode);
    let back = decompress(&arc);
    back == data
}
fn selftest() {
    let mut fails = 0;
    // deterministic pseudo-random via sha256 chain
    let mut rnd = Vec::new();
    let mut s = sha256(b"OMEGA-EXPAND-RANDOM");
    while rnd.len() < 20000 { rnd.extend_from_slice(&s); s = sha256(&s); }
    rnd.truncate(20000);
    let repeated = vec![0x41u8; 20000];
    let mut mixed = Vec::new();
    for i in 0..30000u32 { mixed.push((i.wrapping_mul(2654435761) >> 13) as u8); }
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("one_byte", vec![0x5A]),
        ("random20k", rnd),
        ("repeated20k", repeated),
        ("structured30k", mixed),
    ];
    for &dirs in &[8usize, 40] {
        for mode in [DirMode::Expand] {
            for (name, data) in &cases {
                let ok = rt_ok(data, dirs, mode);
                println!("SELFTEST|case={}|dirs={}|dir_mode={}|bytes={}|roundtrip={}", name, dirs, dirmode_name(mode), data.len(), ok as u32);
                if !ok { fails += 1; }
            }
        }
    }
    // dup + shuffle modes lossless on a structured case
    for mode in [DirMode::Dup, DirMode::Shuffle] {
        let data = &cases[4].1;
        let ok = rt_ok(data, 20, mode);
        println!("SELFTEST|case=mode_{}|dirs=20|bytes={}|roundtrip={}", dirmode_name(mode), data.len(), ok as u32);
        if !ok { fails += 1; }
    }
    print_omega();
    if fails == 0 { println!("SELFTEST_PASS|all_roundtrips_exact"); }
    else { println!("SELFTEST_FAIL|fails={}", fails); exit(1); }
}

// ===========================================================================
// CLI.
// ===========================================================================
fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a String> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1))
}
fn parse_dirs(a: &[String]) -> usize {
    flag(a, "--dirs").and_then(|s| s.parse::<usize>().ok()).unwrap_or(40).min(NDIR)
}
fn parse_mode(a: &[String]) -> DirMode {
    match flag(a, "--dir-mode").map(|s| s.as_str()) {
        Some("dup") => DirMode::Dup,
        Some("shuffle") => DirMode::Shuffle,
        _ => DirMode::Expand,
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: hutter_omega_expand <verify|compress|decompress|curve|selftest> <in> [out] [--dirs K] [--dir-mode M]");
        exit(2);
    }
    match a[1].as_str() {
        "selftest" => selftest(),
        "verify" => {
            let dirs = parse_dirs(&a); let mode = parse_mode(&a);
            let data = fs::read(&a[2]).expect("read input");
            let arc = compress(&data, dirs, mode);
            let back = decompress(&arc);
            let ok = back == data;
            report("verify", data.len(), arc.len(), ok as u32, dirs, mode);
            if !ok { eprintln!("ROUNDTRIP_FAIL"); exit(1); }
            println!("ROUNDTRIP_OK");
        }
        "compress" => {
            let dirs = parse_dirs(&a); let mode = parse_mode(&a);
            let data = fs::read(&a[2]).expect("read input");
            let arc = compress(&data, dirs, mode);
            fs::write(&a[3], &arc).expect("write archive");
            report("compress", data.len(), arc.len(), 0, dirs, mode);
        }
        "decompress" => {
            let arc = fs::read(&a[2]).expect("read archive");
            let out = decompress(&arc);
            fs::write(&a[3], &out).expect("write output");
            println!("DECOMPRESS|archive_bytes={}|output_bytes={}", arc.len(), out.len());
        }
        "curve" => {
            let data = fs::read(&a[2]).expect("read input");
            curve(&data);
        }
        _ => { eprintln!("unknown mode"); exit(2); }
    }
}
