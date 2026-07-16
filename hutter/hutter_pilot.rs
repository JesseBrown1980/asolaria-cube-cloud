// hutter_pilot.rs — dep-free lossless codec, Hutter pre-test BASE.
// Binary arithmetic coder (fpaq0-derived, proven-symmetric) + order-2 exact
// adaptive bit model. Purpose: produce a REAL measured (archive+decoder) size
// on enwik8 with byte-exact roundtrip. NOT a record claim — a first honest number.
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

const NCTX: usize = 1 << 24; // order-2 exact: (prev2<<8 | prev1)<<8 | node  (node 1..255)

struct Model { t: Vec<u16>, c1: u32, c2: u32 }
impl Model {
    fn new() -> Self { Model { t: vec![2048u16; NCTX], c1: 0, c2: 0 } }
    #[inline]
    fn idx(&self, node: u32) -> usize {
        ((((self.c2 << 8) | self.c1) << 8) | node) as usize & (NCTX - 1)
    }
    #[inline]
    fn p(&self, node: u32) -> u32 { self.t[self.idx(node)] as u32 } // 12-bit P(bit=1), 0..4095
    #[inline]
    fn update(&mut self, node: u32, bit: u32) {
        let i = self.idx(node);
        let pr = self.t[i] as i32;
        let target = (bit as i32) << 12; // 0 or 4096
        self.t[i] = (pr + ((target - pr) >> 5)) as u16;
    }
    #[inline]
    fn push_byte(&mut self, b: u8) { self.c2 = self.c1; self.c1 = b as u32; }
}

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
            let p = m.p(node);
            e.encode(bit, p);
            m.update(node, bit);
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
            let p = m.p(node);
            let bit = d.decode(p);
            m.update(node, bit);
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
    if a.len() < 3 { eprintln!("usage: hutter_pilot <verify|compress|decompress> <in> [out]"); exit(2); }
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
