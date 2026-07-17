// omega_omnibit_ladder.rs — the band-ladder cube-prism compressor.
//
// GOAL: compress via the 64->256->1024->4096 band ladder (6/8/10/12-bit) with the
// Omega OmniBit "played in all directions" (reversible A-view / signed-axis
// transforms) and SGRAM entropy coding BETWEEN every band, byte-exact. Then
// MEASURE honestly whether the ladder beats the FLAT SGRAM codec on the SAME input.
//
// REUSE (kept intact):
//   * SGRAM streaming mix = fpaq0 coder + integer logistic order-0..4 mixer
//     (from sgram_mix.rs) as the per-stream entropy coder. In-memory here.
//   * Band cascade quant (from ladder_ref.rs): symbols()/unsymbols() transcode,
//     codebook() SHA-seeded language, quantize()/dequantize() (quotient+residual,
//     exactly reversible), fold() (Omega).
//   * Omega OmniBit directions (from unified_omega.rs): a_ap / a_in reversible
//     A-views (id/rev/xor-delta/rot/nibble/blkrev/evenodd/qprism).
//
// PIPELINE (compress): W0 = input bytes. For each band b in [6,8,10,12]:
//   syms = symbols(W_b, bits) -> spoken = codebook_speak(syms) ->
//   directed = a_ap(dir[b], spoken)   (Omega OmniBit direction, root-seeded) ->
//   q[i]=directed[i]/step, r[i]=directed[i]%step   (exactly reversible quant) ->
//   entropy-code the RESIDUAL stream r (SGRAM); the QUOTIENT stream becomes the
//   working bytes W_{b+1} = pack(q, bits) that "feeds down into the next line".
//   After band 12 the final quotient stream W4 is entropy-coded too.
//   Archive = SGRAM(r0)|SGRAM(r1)|SGRAM(r2)|SGRAM(r3)|SGRAM(W4) + tiny header.
//
// DECODE reverses 12->10->8->6->bytes: dequantize + inverse-codebook +
// inverse-direction at each band, restoring the original bytes + SHA-256.
//
// EVERYTHING (codebooks, direction schedule, Omega, per-band lengths) is
// regenerated from a FIXED ROOT sha256("ASOLARIA-OMEGA-OMNIBIT-LADDER-V1|8467a937cba309f7")
// and from L0 alone -> charged nothing. archive_ratio=NOT_CLAIMED; claims_final_apex=0.

use std::env;
use std::fs;
use std::process::exit;

type R<T> = Result<T, String>;

// ===========================================================================
// SHA-256 (dep-free; identical math to the reference files).
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
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for ch in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes(ch[i*4..i*4+4].try_into().unwrap()); }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh)=(h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let cbh = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(cbh).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a);h[1]=h[1].wrapping_add(b);h[2]=h[2].wrapping_add(c);h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e);h[5]=h[5].wrapping_add(f);h[6]=h[6].wrapping_add(g);h[7]=h[7].wrapping_add(hh);
    }
    let mut o = [0u8; 32];
    for (i, v) in h.iter().enumerate() { o[i*4..i*4+4].copy_from_slice(&v.to_be_bytes()); }
    o
}
fn hex(b: &[u8]) -> String { const H: &[u8;16]=b"0123456789abcdef"; let mut s=String::with_capacity(b.len()*2); for &x in b { s.push(H[(x>>4)as usize]as char); s.push(H[(x&15)as usize]as char);} s }
fn sha_hex(d: &[u8]) -> String { hex(&sha256(d)) }

// ===========================================================================
// Band transcode + pack + codebook + quant + Omega fold (from ladder_ref).
// ===========================================================================
// bytes -> band-bit symbols (MSB-first, last symbol zero-padded)
fn symbols(bytes: &[u8], bits: u32) -> Vec<u16> {
    let mut out = Vec::new(); let mut acc = 0u32; let mut held = 0u32; let mask = (1u32<<bits)-1;
    for &b in bytes { acc=(acc<<8)|b as u32; held+=8; while held>=bits { held-=bits; out.push(((acc>>held)&mask) as u16);} }
    if held>0 { out.push(((acc<<(bits-held))&mask) as u16); }
    out
}
// band-bit symbols -> bytes, truncated to original_len (inverse of symbols for that len)
fn unsymbols(syms: &[u16], bits: u32, original_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(original_len); let mut acc=0u32; let mut held=0u32;
    for &s in syms { acc=(acc<<bits)|(s as u32 & ((1u32<<bits)-1)); held+=bits; while held>=8 { held-=8; out.push(((acc>>held)&0xff) as u8);} }
    out.truncate(original_len); out
}
// pack arbitrary band-bit symbols -> bytes (with tail padding)
fn pack(syms: &[u16], bits: u32) -> Vec<u8> {
    let mut out = Vec::new(); let mut acc=0u32; let mut held=0u32;
    for &s in syms { acc=(acc<<bits)|s as u32; held+=bits; while held>=8 { held-=8; out.push(((acc>>held)&0xff) as u8);} }
    if held>0 { out.push(((acc<<(8-held))&0xff) as u8); }
    out
}
// bytes -> exactly `cnt` band-bit symbols (inverse of pack for that count)
fn unpack(b: &[u8], bits: u32, cnt: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(cnt); let mut acc=0u32; let mut held=0u32; let mask=(1u32<<bits)-1;
    for &x in b { acc=(acc<<8)|x as u32; held+=8; while held>=bits { if out.len()==cnt {break;} held-=bits; out.push(((acc>>held)&mask) as u16);} if out.len()==cnt {break;} }
    out
}
// number of symbols symbols() would produce for a byte-length L at width bits
fn sym_count(l: usize, bits: u32) -> usize { (l*8 + bits as usize - 1) / bits as usize }
// number of bytes pack() would produce for cnt symbols at width bits
fn pack_len(cnt: usize, bits: u32) -> usize { (cnt*bits as usize + 7) / 8 }

// SHA-seeded reversible codebook (never-English permutation of the symbol alphabet)
fn codebook(seed: &[u8; 32], bits: u32) -> (Vec<u16>, Vec<u16>) {
    let size = 1usize << bits;
    let mut keyed: Vec<([u8;32], usize)> = (0..size).map(|value| {
        let mut input = b"OMNIBIT-LADDER-LANGUAGE-V1\0".to_vec();
        input.extend_from_slice(seed);
        input.extend_from_slice(&(value as u64).to_le_bytes());
        (sha256(&input), value)
    }).collect();
    keyed.sort_by(|a,b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut forward = vec![0u16; size];
    for (new,(_,old)) in keyed.into_iter().enumerate() { forward[old]=new as u16; }
    let mut inverse = vec![0u16; size];
    for (old,&new) in forward.iter().enumerate() { inverse[new as usize]=old as u16; }
    (forward, inverse)
}
fn map_symbols(input: &[u16], map: &[u16]) -> Vec<u16> { input.iter().map(|&v| map[v as usize]).collect() }
fn quantize(input: &[u16], step: u16) -> (Vec<u16>, Vec<u16>) {
    (input.iter().map(|&v| v/step).collect(), input.iter().map(|&v| v%step).collect())
}
fn dequantize(buckets: &[u16], residuals: &[u16], step: u16) -> Vec<u16> {
    buckets.iter().zip(residuals).map(|(&q,&r)| q*step + r).collect()
}
fn fold(domain: &str, parent: &str, leaves: &[String]) -> String {
    let mut sorted = leaves.to_vec(); sorted.sort();
    let mut input = domain.as_bytes().to_vec(); input.push(0); input.extend_from_slice(parent.as_bytes());
    for leaf in sorted { input.extend_from_slice(&(leaf.len() as u32).to_le_bytes()); input.extend_from_slice(leaf.as_bytes()); }
    sha_hex(&input)
}

// ===========================================================================
// Omega OmniBit directions (a_ap / a_in), from unified_omega.rs — reversible.
// The qprism/blkrev seeds come from the ROOT (data-independent) so the decoder
// regenerates the exact same permutation. Verbatim transform math.
// ===========================================================================
fn revb(v: u16, n: u32) -> u16 { let mut o=0; for i in 0..n { if v&(1<<i)!=0 { o|=1<<(n-1-i);} } o }
fn g_r(d: &[u16]) -> Vec<u16> { d.iter().rev().copied().collect() }
fn g_n(d: &[u16], bits: u32) -> Vec<u16> { let h=bits/2; let hm=(1u16<<h)-1; d.iter().map(|&g| ((g&hm)<<h)|(g>>h)).collect() }
fn rol(g: u16, bits: u32) -> u16 { let m=(1u16<<bits)-1; ((g<<1)|(g>>(bits-1)))&m }
fn ror(g: u16, bits: u32) -> u16 { let m=(1u16<<bits)-1; ((g>>1)|((g&1)<<(bits-1)))&m }
fn blkrev(d: &[u16], bl: usize) -> Vec<u16> { let mut o=Vec::new(); for c in d.chunks(bl) { o.extend(c.iter().rev()); } o }
fn xd(d: &[u16]) -> Vec<u16> { if d.is_empty() {return vec![];} let mut o=vec![d[0]]; for i in 1..d.len() { o.push(d[i]^d[i-1]); } o }
fn xu(d: &[u16]) -> Vec<u16> { if d.is_empty() {return vec![];} let mut o=vec![d[0]]; for i in 1..d.len() { let b=d[i]^o[i-1]; o.push(b); } o }
fn evo(d: &[u16]) -> Vec<u16> { d.iter().step_by(2).chain(d.iter().skip(1).step_by(2)).copied().collect() }
fn uevo(d: &[u16]) -> Vec<u16> { let e=(d.len()+1)/2; let mut o=vec![0u16;d.len()]; for i in 0..e {o[i*2]=d[i];} for i in e..d.len(){o[(i-e)*2+1]=d[i];} o }
fn qpo(nb: usize, ss: &[u8;32]) -> Vec<usize> { let mut k:Vec<([u8;32],usize)>=(0..nb).map(|i|{let mut s=b"QPRISM_LADDER_V1|".to_vec();s.extend_from_slice(ss);s.extend_from_slice(&(i as u64).to_le_bytes());(sha256(&s),i)}).collect(); k.sort_by(|a,b|a.0.cmp(&b.0).then(a.1.cmp(&b.1))); k.into_iter().map(|x|x.1).collect() }
fn qp(d: &[u16], ss: &[u8;32]) -> Vec<u16> { let bl=257; let n=d.len()/bl; let o=qpo(n,ss); let mut r=Vec::new(); for i in o {r.extend_from_slice(&d[i*bl..(i+1)*bl]);} r.extend_from_slice(&d[n*bl..]); r }
fn uqp(d: &[u16], ss: &[u8;32]) -> Vec<u16> { let bl=257; let n=d.len()/bl; let o=qpo(n,ss); let mut r=vec![0u16;d.len()]; for (p,i) in o.into_iter().enumerate(){r[i*bl..(i+1)*bl].copy_from_slice(&d[p*bl..(p+1)*bl]);} r[n*bl..].copy_from_slice(&d[n*bl..]); r }
const A_NAMES: [&str; 8] = ["G_ID","G_REV","G_XORD","G_ROT","G_HALF","G_BLKREV","G_EVODD","G_QPRISM"];
fn a_ap(ix: usize, d: &[u16], bits: u32, ss: &[u8;32]) -> Vec<u16> {
    match ix { 0=>d.to_vec(),1=>g_r(d),2=>xd(d),3=>d.iter().rev().map(|&g|rol(g,bits)).collect(),4=>g_n(d,bits),5=>blkrev(d,256),6=>evo(d),7=>qp(d,ss),_=>unreachable!() }
}
fn a_in(ix: usize, d: &[u16], bits: u32, ss: &[u8;32]) -> Vec<u16> {
    match ix { 0=>d.to_vec(),1=>g_r(d),2=>xu(d),3=>d.iter().map(|&g|ror(g,bits)).rev().collect(),4=>g_n(d,bits),5=>blkrev(d,256),6=>uevo(d),7=>uqp(d,ss),_=>unreachable!() }
}

// ===========================================================================
// SGRAM entropy coder (order-0..4 logistic mix + fpaq0 AC), IN-MEMORY.
// Model math + coder math copied VERBATIM from sgram_mix.rs; only the byte
// sink/source is a Vec<u8>/slice. Bytes emitted are identical to the streaming
// coder. sgram_compress(x) is the flat baseline; used per-band in the ladder.
// ===========================================================================
#[inline]
fn squash(d: i32) -> i32 {
    const T: [i32; 33] = [1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047,2549,2994,3348,3607,3785,3901,3975,4022,4050,4068,4079,4085,4089,4092,4093,4094];
    if d > 2047 { return 4095; }
    if d < -2047 { return 0; }
    let w = d & 127; let idx = ((d>>7)+16) as usize;
    (T[idx]*(128-w) + T[idx+1]*w + 64) >> 7
}
fn build_stretch() -> Vec<i32> {
    let mut stretch = vec![0i32; 4096]; let mut pi: i32 = 0;
    for x in -2047..=2047 { let p = squash(x); while pi<=p { stretch[pi as usize]=x; pi+=1; } }
    while (pi as usize) < 4096 { stretch[pi as usize]=2047; pi+=1; }
    stretch
}
const NIN: usize = 6; const O0SIZE: usize = 256; const O1SIZE: usize = 1<<16; const HSIZE: usize = 1<<22;
const MC: usize = 256; const RATE: i32 = 4; const LR_SHIFT: i32 = 12; const WCLAMP: i32 = 1<<24;
struct Model { t0:Vec<u16>,t1:Vec<u16>,t2:Vec<u16>,t3:Vec<u16>,t4:Vec<u16>,w:Vec<i32>,c1:u32,c2:u32,c3:u32,c4:u32,h2:u32,h3:u32,h4:u32,idx:[usize;5],st:[i32;NIN],wctx:usize,pr:i32,stretch:Vec<i32> }
#[inline]
fn hidx(h: u32, node: u32, size: usize) -> usize { let x=(h ^ node.wrapping_mul(0x9E37_79B1)).wrapping_mul(0x2545_F491); (x as usize)&(size-1) }
impl Model {
    fn new() -> Self {
        let mut w = vec![0i32; MC*NIN]; let init = (65536/5) as i32;
        for c in 0..MC { for i in 0..5 { w[c*NIN+i]=init; } }
        Model { t0:vec![2048u16;O0SIZE],t1:vec![2048u16;O1SIZE],t2:vec![2048u16;HSIZE],t3:vec![2048u16;HSIZE],t4:vec![2048u16;HSIZE],w,c1:0,c2:0,c3:0,c4:0,h2:0,h3:0,h4:0,idx:[0;5],st:[0;NIN],wctx:0,pr:2048,stretch:build_stretch() }
    }
    #[inline]
    fn predict(&mut self, node: u32) -> u32 {
        self.idx[0]=(node as usize)&(O0SIZE-1);
        self.idx[1]=(((self.c1<<8)|node) as usize)&(O1SIZE-1);
        self.idx[2]=hidx(self.h2,node,HSIZE); self.idx[3]=hidx(self.h3,node,HSIZE); self.idx[4]=hidx(self.h4,node,HSIZE);
        let p0=self.t0[self.idx[0]] as usize; let p1=self.t1[self.idx[1]] as usize; let p2=self.t2[self.idx[2]] as usize; let p3=self.t3[self.idx[3]] as usize; let p4=self.t4[self.idx[4]] as usize;
        self.st[0]=self.stretch[p0]; self.st[1]=self.stretch[p1]; self.st[2]=self.stretch[p2]; self.st[3]=self.stretch[p3]; self.st[4]=self.stretch[p4]; self.st[5]=256;
        self.wctx=(self.c1 as usize)&(MC-1); let base=self.wctx*NIN;
        let mut dot: i64 = 0; for i in 0..NIN { dot += (self.st[i] as i64)*(self.w[base+i] as i64); }
        let mut d=(dot>>16) as i32; if d>2047 {d=2047;} if d < -2047 {d=-2047;}
        let mut p=squash(d); if p<1 {p=1;} if p>4094 {p=4094;} self.pr=p; p as u32
    }
    #[inline]
    fn update(&mut self, bit: u32) {
        let target=(bit as i32)<<12;
        { let t=&mut self.t0[self.idx[0]]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }
        { let t=&mut self.t1[self.idx[1]]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }
        { let t=&mut self.t2[self.idx[2]]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }
        { let t=&mut self.t3[self.idx[3]]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }
        { let t=&mut self.t4[self.idx[4]]; let pr=*t as i32; *t=(pr+((target-pr)>>RATE)) as u16; }
        let err=target-self.pr; let base=self.wctx*NIN;
        for i in 0..NIN { let mut nw=self.w[base+i]+((self.st[i]*err)>>LR_SHIFT); if nw>WCLAMP {nw=WCLAMP;} if nw < -WCLAMP {nw=-WCLAMP;} self.w[base+i]=nw; }
    }
    #[inline]
    fn push_byte(&mut self, b: u8) {
        self.c4=self.c3; self.c3=self.c2; self.c2=self.c1; self.c1=b as u32;
        self.h2=self.c1.wrapping_mul(0x6B43_A9B5).wrapping_add(self.c2.wrapping_add(1).wrapping_mul(0x9E37_79B1));
        self.h3=self.h2.wrapping_mul(0x2545_F491).wrapping_add(self.c3.wrapping_add(1).wrapping_mul(0x85EB_CA77));
        self.h4=self.h3.wrapping_mul(0x2545_F491).wrapping_add(self.c4.wrapping_add(1).wrapping_mul(0xC2B2_AE35));
    }
}
struct Enc { x1:u32, x2:u32, out:Vec<u8> }
impl Enc {
    fn new() -> Self { Enc{x1:0,x2:0xFFFF_FFFF,out:Vec::new()} }
    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
        let range=self.x2-self.x1; let xmid=self.x1+(range>>12)*p;
        if bit==1 { self.x2=xmid; } else { self.x1=xmid+1; }
        while (self.x1^self.x2)&0xFF00_0000==0 { self.out.push((self.x2>>24) as u8); self.x1<<=8; self.x2=(self.x2<<8)|0xFF; }
    }
    fn flush(mut self) -> Vec<u8> { for _ in 0..4 { self.out.push((self.x1>>24) as u8); self.x1<<=8; } self.out }
}
struct Dec<'a> { x1:u32, x2:u32, x:u32, d:&'a[u8], pos:usize }
impl<'a> Dec<'a> {
    fn new(d:&'a[u8]) -> Self { let mut de=Dec{x1:0,x2:0xFFFF_FFFF,x:0,d,pos:0}; for _ in 0..4 { de.x=(de.x<<8)|de.next() as u32; } de }
    #[inline]
    fn next(&mut self) -> u8 { if self.pos>=self.d.len() { return 0; } let b=self.d[self.pos]; self.pos+=1; b }
    #[inline]
    fn decode(&mut self, p: u32) -> u32 {
        let range=self.x2-self.x1; let xmid=self.x1+(range>>12)*p;
        let bit=if self.x<=xmid {1} else {0};
        if bit==1 { self.x2=xmid; } else { self.x1=xmid+1; }
        while (self.x1^self.x2)&0xFF00_0000==0 { self.x1<<=8; self.x2=(self.x2<<8)|0xFF; self.x=(self.x<<8)|self.next() as u32; }
        bit
    }
}
fn sgram_compress(data: &[u8]) -> Vec<u8> {
    let mut m = Model::new(); let mut e = Enc::new();
    for &byte in data {
        let mut node: u32 = 1;
        for i in (0..8).rev() { let bit=((byte>>i)&1) as u32; let p=m.predict(node); e.encode(bit,p); m.update(bit); node=(node<<1)|bit; }
        m.push_byte(byte);
    }
    e.flush()
}
fn sgram_decompress(arc: &[u8], n: usize) -> Vec<u8> {
    let mut m = Model::new(); let mut d = Dec::new(arc); let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut node: u32 = 1;
        for _ in 0..8 { let p=m.predict(node); let bit=d.decode(p); m.update(bit); node=(node<<1)|bit; }
        let byte=(node&0xFF) as u8; out.push(byte); m.push_byte(byte);
    }
    out
}

// ===========================================================================
// The band-ladder cube-prism.
// ===========================================================================
const ROOT_LABEL: &str = "ASOLARIA-OMEGA-OMNIBIT-LADDER-V1|8467a937cba309f7";
// (bits, step) for the 64/256/1024/4096 bands.
const BANDS: [(u32, u16); 4] = [(6,2),(8,4),(10,8),(12,16)];
const BAND_NAMES: [&str; 4] = ["BEHCS64","BEHCS256","BEHCS1024","BEHCS4096"];

fn root() -> [u8; 32] { sha256(ROOT_LABEL.as_bytes()) }
// per-band codebook seed, direction index, and qprism seed — all from the ROOT.
fn band_cb_seed(rt: &[u8;32], bits: u32) -> [u8;32] { let mut s=b"BAND-CODEBOOK|".to_vec(); s.extend_from_slice(rt); s.extend_from_slice(&bits.to_le_bytes()); sha256(&s) }
fn band_dir(rt: &[u8;32], bi: usize) -> usize { (rt[bi] as usize) % 8 }
fn band_dir_seed(rt: &[u8;32], bits: u32) -> [u8;32] { let mut s=b"BAND-DIRECTION|".to_vec(); s.extend_from_slice(rt); s.extend_from_slice(&bits.to_le_bytes()); sha256(&s) }

// varint-free fixed header: magic + L0 + sha32 + 5*(comp_len u64, raw_len u64)
const MAGIC: &[u8;4] = b"OOL1";

struct BandReport { name: String, dir: String, floor_omega: String, residual_bytes: usize, residual_comp: usize }

fn compress(input: &[u8]) -> (Vec<u8>, Vec<BandReport>, String) {
    let rt = root();
    let src_sha = sha256(input);
    let mut w = input.to_vec();
    let mut segments: Vec<(Vec<u8>, usize)> = Vec::new(); // (compressed, raw_len)
    let mut reports = Vec::new();
    let mut floor_leaves = Vec::new();

    for (bi, &(bits, step)) in BANDS.iter().enumerate() {
        let (fwd, _inv) = codebook(&band_cb_seed(&rt, bits), bits);
        let dir = band_dir(&rt, bi);
        let dseed = band_dir_seed(&rt, bits);
        let syms = symbols(&w, bits);
        let spoken = map_symbols(&syms, &fwd);
        let directed = a_ap(dir, &spoken, bits, &dseed);
        let (q, r) = quantize(&directed, step);
        let r_bytes = pack(&r, bits);
        let r_comp = sgram_compress(&r_bytes);
        // FLOOROMEGA per band: commit the band's residual + quotient leaves.
        let floor = fold("OMNIBIT-FLOOR-V1", &sha_hex(&r_bytes), &[sha_hex(&pack(&q, bits))]);
        reports.push(BandReport { name: BAND_NAMES[bi].into(), dir: A_NAMES[dir].into(), floor_omega: floor.clone(), residual_bytes: r_bytes.len(), residual_comp: r_comp.len() });
        floor_leaves.push(floor);
        segments.push((r_comp, r_bytes.len()));
        w = pack(&q, bits); // quotient feeds down into the next band
    }
    // final quotient stream (the deepest carry) entropy-coded as segment 5
    let w_comp = sgram_compress(&w);
    segments.push((w_comp, w.len()));

    let unified = fold("OMNIBIT-UNIFIED-OMEGA-V1", &hex(&src_sha), &floor_leaves);

    // assemble archive
    let mut arc = Vec::new();
    arc.extend_from_slice(MAGIC);
    arc.extend_from_slice(&(input.len() as u64).to_le_bytes());
    arc.extend_from_slice(&src_sha);
    for (c, raw) in &segments { arc.extend_from_slice(&(c.len() as u64).to_le_bytes()); arc.extend_from_slice(&(*raw as u64).to_le_bytes()); }
    for (c, _) in &segments { arc.extend_from_slice(c); }
    (arc, reports, unified)
}

fn decompress(arc: &[u8]) -> R<Vec<u8>> {
    if arc.len() < 4+8+32+5*16 { return Err("archive too short".into()); }
    if &arc[..4] != MAGIC { return Err("bad magic".into()); }
    let l0 = u64::from_le_bytes(arc[4..12].try_into().unwrap()) as usize;
    let src_sha: [u8;32] = arc[12..44].try_into().unwrap();
    let mut off = 44;
    let mut seg_meta = Vec::new(); // (comp_len, raw_len)
    for _ in 0..5 { let c=u64::from_le_bytes(arc[off..off+8].try_into().unwrap()) as usize; let raw=u64::from_le_bytes(arc[off+8..off+16].try_into().unwrap()) as usize; seg_meta.push((c,raw)); off+=16; }
    let mut segs = Vec::new();
    for &(c,_) in &seg_meta { segs.push(&arc[off..off+c]); off+=c; }

    let rt = root();
    // derive per-band L_b (byte length into band) and nsym_b from L0
    let mut l_in = [0usize;4]; let mut nsym = [0usize;4];
    let mut lb = l0;
    for bi in 0..4 { let (bits,_)=BANDS[bi]; l_in[bi]=lb; nsym[bi]=sym_count(lb,bits); lb=pack_len(nsym[bi],bits); }
    // decode the 5 stored streams
    let r_bytes: Vec<Vec<u8>> = (0..4).map(|bi| sgram_decompress(segs[bi], seg_meta[bi].1)).collect();
    let w4 = sgram_decompress(segs[4], seg_meta[4].1);

    // reverse the ladder: 12 -> 10 -> 8 -> 6 -> bytes
    let mut w_next = w4; // = pack(q3, 12bits)
    for bi in (0..4).rev() {
        let (bits, step) = BANDS[bi];
        let (_fwd, inv) = codebook(&band_cb_seed(&rt, bits), bits);
        let dir = band_dir(&rt, bi);
        let dseed = band_dir_seed(&rt, bits);
        let q = unpack(&w_next, bits, nsym[bi]);
        let r = unpack(&r_bytes[bi], bits, nsym[bi]);
        if q.len()!=nsym[bi] || r.len()!=nsym[bi] { return Err(format!("band {} length mismatch q={} r={} want {}", bi, q.len(), r.len(), nsym[bi])); }
        let directed = dequantize(&q, &r, step);
        let spoken = a_in(dir, &directed, bits, &dseed);
        let syms = map_symbols(&spoken, &inv);
        w_next = unsymbols(&syms, bits, l_in[bi]);
    }
    if w_next.len()!=l0 { return Err(format!("final length {} != {}", w_next.len(), l0)); }
    if sha256(&w_next)!=src_sha { return Err("sha mismatch after restore".into()); }
    Ok(w_next)
}

fn decoder_src_bytes() -> u64 { fs::metadata(file!()).map(|m| m.len()).unwrap_or(0) }

// ===========================================================================
// measurement + roundtrip
// ===========================================================================
fn flat_sgram_archive(input: &[u8]) -> usize {
    // baseline archive = 8-byte length header + AC stream (as sgram_mix on disk).
    8 + sgram_compress(input).len()
}

fn measure(input: &[u8], label: &str) -> R<(usize, usize, bool, bool, String)> {
    let (arc, reports, unified) = compress(input);
    let restored = decompress(&arc)?;
    let exact = restored == input;
    let flat = flat_sgram_archive(input);
    let ladder = arc.len();
    let compounds = ladder < flat;
    let bpc = if input.is_empty() {0.0} else {(ladder as f64 * 8.0)/input.len() as f64};
    for r in &reports {
        eprintln!("  BAND|name={}|dir={}|residual_raw={}|residual_comp={}|floor_omega={}", r.name, r.dir, r.residual_bytes, r.residual_comp, &r.floor_omega[..16]);
    }
    eprintln!("  UNIFIEDOMEGA={}", unified);
    println!(
        "OMNIBIT_LADDER|corpus={}|input_bytes={}|ladder_archive_bytes={}|flat_sgram_bytes={}|compounds={}|bpc={:.4}|roundtrip_exact={}|decoder_src_bytes={}|archive_ratio=NOT_CLAIMED|claims_final_apex=0|json=0",
        label, input.len(), ladder, flat, compounds as u32, bpc, exact as u32, decoder_src_bytes()
    );
    Ok((ladder, flat, compounds, exact, unified))
}

fn selftest() -> R<()> {
    // edge cases: empty / 1-byte / repeated / random / structured text
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    cases.push(("empty", vec![]));
    cases.push(("one_byte", vec![0x41]));
    cases.push(("one_zero", vec![0x00]));
    cases.push(("repeated", vec![0x61u8; 5000]));
    cases.push(("two_bytes", vec![0xff, 0x00]));
    // pseudo-random
    let mut rnd = Vec::new(); let mut s = sha256(b"OMNIBIT-RANDOM-SEED");
    while rnd.len() < 40000 { rnd.extend_from_slice(&s); s = sha256(&s); }
    rnd.truncate(37913); cases.push(("random_37913", rnd));
    // structured english-ish
    let mut txt = Vec::new(); let phrase = b"the quick brown fox jumps over the lazy dog. ";
    while txt.len() < 60000 { txt.extend_from_slice(phrase); }
    txt.truncate(58321); cases.push(("english_58321", txt));

    let mut all_ok = true;
    for (name, data) in &cases {
        let (arc, _r, _u) = compress(data);
        let restored = decompress(&arc)?;
        let ok = &restored == data;
        if !ok { all_ok = false; }
        println!("SELFTEST|case={}|input={}|archive={}|roundtrip_exact={}|json=0", name, data.len(), arc.len(), ok as u32);
        if !ok { return Err(format!("roundtrip FAILED on case {}", name)); }
    }
    // also confirm bands are reversible independently: quant + direction + codebook
    for &(bits, step) in &BANDS.iter().copied().collect::<Vec<_>>() {
        let rt = root(); let (fwd, inv) = codebook(&band_cb_seed(&rt, bits), bits); let dseed = band_dir_seed(&rt, bits);
        let data = &cases[6].1; // english
        let syms = symbols(data, bits);
        for dir in 0..8usize {
            let spoken = map_symbols(&syms, &fwd);
            let directed = a_ap(dir, &spoken, bits, &dseed);
            let (q, r) = quantize(&directed, step);
            let deq = dequantize(&q, &r, step);
            let back_spoken = a_in(dir, &deq, bits, &dseed);
            let back_syms = map_symbols(&back_spoken, &inv);
            if back_syms != syms || unsymbols(&back_syms, bits, data.len()) != *data {
                return Err(format!("band bits={} dir={} not reversible", bits, dir));
            }
        }
    }
    println!("BANDS_REVERSIBLE|all_bits=6,8,10,12|all_dirs=0..8|exact=1|json=0");
    if all_ok { println!("SELFTEST_PASS|json=0"); Ok(()) } else { Err("selftest failed".into()) }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 2 { eprintln!("usage: omega_omnibit_ladder <compress|decompress|verify|pilot|selftest> ..."); exit(2); }
    match a[1].as_str() {
        "selftest" => { if let Err(e)=selftest() { eprintln!("ERROR: {}", e); exit(1); } }
        "compress" => {
            if a.len()<4 { eprintln!("usage: compress <in> <archive>"); exit(2); }
            let input = fs::read(&a[2]).expect("read input");
            let (arc,_,_) = compress(&input);
            fs::write(&a[3], &arc).expect("write archive");
            println!("COMPRESS|input_bytes={}|archive_bytes={}|json=0", input.len(), arc.len());
        }
        "decompress" => {
            if a.len()<4 { eprintln!("usage: decompress <archive> <out>"); exit(2); }
            let arc = fs::read(&a[2]).expect("read archive");
            match decompress(&arc) { Ok(out)=>{ fs::write(&a[3], &out).expect("write out"); println!("DECOMPRESS|out_bytes={}|json=0", out.len()); } Err(e)=>{ eprintln!("ERROR: {}", e); exit(1); } }
        }
        "verify" | "pilot" => {
            if a.len()<3 { eprintln!("usage: verify <in>"); exit(2); }
            let input = fs::read(&a[2]).expect("read input");
            match measure(&input, &a[2]) { Ok((_,_,_,exact,_))=>{ if !exact { eprintln!("ROUNDTRIP_FAIL"); exit(1); } } Err(e)=>{ eprintln!("ERROR: {}", e); exit(1); } }
        }
        _ => { eprintln!("unknown mode"); exit(2); }
    }
}
