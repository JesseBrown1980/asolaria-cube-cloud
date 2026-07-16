#!/usr/bin/env bash
# Build + roundtrip-test + learning-curve for the key="dir" Hutter pilot variant.
set -u
export PATH="$HOME/.cargo/bin:$PATH"

DIR=/mnt/c/Users/acer/AppData/Local/Temp/claude/C--Users-acer/2e472baa-abf3-4d17-b87f-f0720b3f4815/scratchpad/cube-cloud/hutter/variants/dir
WORK=/tmp/hp_dir_work
mkdir -p "$WORK"

echo "== rustc version =="
rustc --version || { echo "NO_RUSTC"; exit 3; }

echo "== compile =="
rustc --edition=2021 -O "$DIR/hp_dir.rs" -o "$WORK/hp_dir" 2>&1 || { echo "COMPILE_FAIL"; exit 4; }
echo "COMPILE_OK"
BIN="$WORK/hp_dir"

# ---- Build ~1.5MB English text corpus from local .md docs ----
SRC="$WORK/src.txt"
: > "$SRC"
cat /mnt/c/Users/acer/CLAUDE.md >> "$SRC" 2>/dev/null || true
for f in /mnt/c/Users/acer/.claude/projects/C--/memory/*.md; do
  [ -f "$f" ] && cat "$f" >> "$SRC" 2>/dev/null || true
done
SRCSZ=$(stat -c%s "$SRC" 2>/dev/null || echo 0)
echo "== source md bytes: $SRCSZ =="

CORP="$WORK/corpus.txt"
: > "$CORP"
if [ "$SRCSZ" -lt 1000 ]; then
  # Fallback: synthesize English-ish text so the test still runs.
  python3 - "$CORP" <<'PY'
import sys, random
random.seed(7)
words=("the of and to in a is that it was for on are as with his they at be this from "
       "have or by one had not but what all were when we there can an your which their said "
       "asolaria fabric cube omega compression codec arithmetic model direction heldout passes "
       "learning curve entropy context order stride mixer weight prediction byte stream").split()
out=open(sys.argv[1],"w")
n=0
while n<1500000:
    line=" ".join(random.choice(words) for _ in range(random.randint(6,14)))+".\n"
    out.write(line); n+=len(line)
out.close()
PY
else
  while [ "$(stat -c%s "$CORP")" -lt 1500000 ]; do cat "$SRC" >> "$CORP"; done
fi
head -c 1500000 "$CORP" > "$CORP.trim" && mv "$CORP.trim" "$CORP"
CORPSZ=$(stat -c%s "$CORP")
echo "== corpus bytes: $CORPSZ =="

# ---- gzip -9 baseline ----
GZSZ=$(gzip -9 -c "$CORP" | wc -c)
echo "GZIP9|corpus_bytes=$CORPSZ|gzip9_bytes=$GZSZ"

# ---- Edge-case inputs ----
RAND="$WORK/rand.bin";  head -c 512000 /dev/urandom > "$RAND"
EMPTY="$WORK/empty.bin"; : > "$EMPTY"
ONE="$WORK/one.bin";     printf 'A' > "$ONE"
REP="$WORK/rep.bin";     python3 -c "open('$REP','wb').write(b'AB'*20)"   # 40 bytes, repeated

echo "== ROUNDTRIP VERIFY =="
echo "-- corpus passes=1 (expand) --"; "$BIN" verify "$CORP" --passes 1 --dir-mode expand
echo "-- corpus passes=4 (expand) --"; "$BIN" verify "$CORP" --passes 4 --dir-mode expand
echo "-- corpus passes=4 (dup)    --"; "$BIN" verify "$CORP" --passes 4 --dir-mode dup
echo "-- corpus passes=4 (shuffle)--"; "$BIN" verify "$CORP" --passes 4 --dir-mode shuffle
echo "-- random 500KB passes=4 (expand) --"; "$BIN" verify "$RAND" --passes 4 --dir-mode expand
echo "-- empty passes=4 --";  "$BIN" verify "$EMPTY" --passes 4 --dir-mode expand
echo "-- 1-byte passes=4 --"; "$BIN" verify "$ONE"   --passes 4 --dir-mode expand
echo "-- 40-byte repeated passes=4 --"; "$BIN" verify "$REP" --passes 4 --dir-mode expand
echo "-- 40-byte repeated passes=1 --"; "$BIN" verify "$REP" --passes 1 --dir-mode expand

echo "== FILE COMPRESS/DECOMPRESS ROUNDTRIP (separate invocations) =="
"$BIN" compress "$CORP" "$WORK/corpus.arc" --passes 4 --dir-mode expand
"$BIN" decompress "$WORK/corpus.arc" "$WORK/corpus.back"
if cmp -s "$CORP" "$WORK/corpus.back"; then echo "FILE_ROUNDTRIP|passes=4|dir_mode=expand|exact=1"; else echo "FILE_ROUNDTRIP|exact=0"; fi

echo "== LEARNING CURVE (heldout-only score) =="
"$BIN" curve "$CORP"

echo "== DONE =="
