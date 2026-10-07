# recapcut — narration ke hisaab se movie clips apne aap

Aap script likho, AI voice se bulwao, aur poori movie do. **recapcut** narration ki har line ke
liye movie ka sahi scene dhoondhta hai, clip ko utna hi lamba kaat-ta hai jitni line hai, aur
sab jod kar final video bana deta hai. Video aur narration frame-exact sync me rehte hain.

```
script.txt + voice.mp3 + movie.mp4 (+ movie.srt)  ──►  recap.mp4  +  plan.csv (edit karne layak)
```

## Ye kaise kaam karta hai

1. **Narration timing.** Script ki har line audio me kab shuru aur kab khatam hoti hai:
   - script + audio + Whisper: word-level timing (sabse accurate)
   - script + audio (Whisper nahi): AI voice ke pauses se line breaks
   - sirf script: tool khud free AI voice (edge-tts) banata hai, timing exact
2. **Movie ke shots.** ffmpeg se har scene cut dhoondhta hai (cache hota hai, dobara nahi chalta).
3. **Har line ke liye har shot ka score:**
   - **Visual (CLIP):** shot dikhne me line jaisa hai? ("baarish me rona", "car chase")
     Hindi/Hinglish text bhi chalta hai.
   - **Dialogue:** shot ke aas-paas ke subtitles line se milte hain? (naam, key words)
   - **Claude AI (`--ai`, optional):** Claude poore subtitles padh kar batata hai ki line movie
     me kahan hai. Movie ka plot jaanta ho to aur behtar.
   - **Story order:** recap aksar movie ke order me chalta hai.
   - **Aapke hints:** script me `[01:12:30]` likh do, wo line wahin se aayegi.
4. **Best sequence.** Ek DP algorithm har line ke liye shot chunta hai: total score max, piche
   jaane par penalty, same shot repeat par penalty.
5. **Render.** Har clip frame-exact kaati jaati hai, 16:9 ya 9:16 (Shorts) me, aur upar narration.

## Install

1. **Python 3.10+** — https://www.python.org/downloads/ (Windows par "Add to PATH" tick karo)
2. **ffmpeg** — Windows: `winget install Gyan.FFmpeg` · Mac: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg`
3. Is folder me:

```bash
pip install -r requirements.txt        # basic mode
pip install -r requirements-ai.txt     # full accuracy (Whisper + CLIP + AI voice + Claude)
pip install -r requirements-app.txt    # browser app
python -m recapcut check               # kya kya chalu hai, dikhata hai
```

NVIDIA GPU ho to `requirements-ai.txt` se pehle CUDA wala torch install karo
(https://pytorch.org/get-started/locally/). CPU par bhi chalta hai. Pehli baar poori movie scan hoti hai,
jisme movie aur PC ke hisaab se 10-30 minute lag sakte hain. Uske baad wahi movie cache se
turant chalti hai.

## App (sabse aasaan tareeka)

**Windows:** `Start-App.bat` par double-click karo.
**Mac / Linux:** `./start_app.sh` chalao.

Pehli baar packages install honge. Phir browser me **http://127.0.0.1:7860** khulega:

1. Movie upload karo (ya badi file ho to uska path paste karo, jaise `D:\Movies\movie.mkv`)
2. Subtitles `.srt` daalo (optional, par accuracy badhti hai)
3. Apni AI voice ka audio daalo aur script paste karo
4. **🎬 Video Banao** dabao, progress live dikhega, video wahin play hogi
5. Neeche table me koi clip galat lage to uska `movie_time` badlo aur **🔁 dobara render** dabao

Videos `~/recapcut_output/` (Windows: `C:\Users\<naam>\recapcut_output`) me save hoti hain.
Command line se: `python -m recapcut app`. Phone par kholna ho to `python -m recapcut app --share`
(temporary public link banata hai; movie aapke PC par hi process hoti hai).

## Command line se use

### 1) Aapki script + aapki AI voice (recommended)

```bash
python -m recapcut make --movie movie.mp4 --audio voice.mp3 --script script.txt --subs movie.srt --language hi --out recap.mp4
```

### 2) Sirf script, voice tool khud banaye

```bash
python -m recapcut make --movie movie.mp4 --script script.txt --subs movie.srt --voice hi-IN-MadhurNeural --out recap.mp4
```

Hindi voices: `hi-IN-MadhurNeural` (male), `hi-IN-SwaraNeural` (female). Speed: `--tts-rate +10%`.

### 3) Sirf audio (script nahi)

```bash
python -m recapcut make --movie movie.mp4 --audio voice.mp3 --subs movie.srt --language hi
```

### Sabse zyada accuracy: Claude se timestamps

```bash
set ANTHROPIC_API_KEY=sk-ant-...      # Mac/Linux: export ANTHROPIC_API_KEY=...
python -m recapcut make --movie movie.mp4 --audio voice.mp3 --script script.txt --subs movie.srt --ai --movie-title "Movie Ka Naam (2023)"
```

Claude poore subtitles ek baar padhta hai. 2 ghante ki movie ka ek run lagbhag $0.10–0.30 ka
padta hai (Claude Opus 5.5). Model badalna ho to `--ai-model claude-sonnet-5-5`.

### YouTube Shorts / Reels (9:16)

```bash
python -m recapcut make ... --format portrait            # blur background
python -m recapcut make ... --format portrait --portrait-mode crop
```

## Script format

```text
# '#' wali line comment hai
Raju ek chhote se gaon me rehta hai.
Ek din usse ek ajeeb chitthi milti hai!
[00:25:10] Station par uski mulaqat Meera se hoti hai.
[01:48:00-01:50:30] Climax me dono aamne saamne aate hain.
```

- Har sentence (`.` `!` `?` `।`) ek alag line maani jaati hai.
- `[hh:mm:ss]` / `[mm:ss]` / `[start-end]` hints optional hain. Jo line galat jagah lage,
  bas usi par hint laga do.
- Script wahi text rakho jo AI voice ne bola hai, taaki timing sahi baithe.
- Lambi line apne aap kai chhote cuts me batt jaati hai (`--max-clip 3`, seconds).

Example: [`examples/script_example.txt`](examples/script_example.txt)

## Koi clip galat lagi? 1 minute me theek karo

Har run `recap_work/plan.csv` banata hai:

| n | narr_start | narr_end | movie_time | score | why | text |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 2.950 | 00:03:12.300 | 2.10 | visual:+1.2 text:+0.8 | Raju ek chhote se gaon... |

Excel / Google Sheets me `movie_time` badlo, save karo, phir sirf render chalao (dobara matching nahi hoti):

```bash
python -m recapcut render --plan recap_work/plan.csv --out recap.mp4
```

Sirf plan dekhna ho, video nahi: `--plan-only`.

## Useful options

| Option | Matlab |
|---|---|
| `--subs movie.srt` | movie ke subtitles. Dialogue matching aur `--ai` ke liye. **Accuracy bahut badhta hai.** |
| `--skip-start 90 --skip-end 420` | shuru ke logos / aakhri credits ko kabhi mat chuno |
| `--max-clip 2.5` | cuts kitne tez badlein (chhota = fast-paced) |
| `--movie-volume 0.08` | movie ki original awaaz halki si peeche |
| `--burn-subs` | narration subtitles video par likho (`recap_work/narration.srt` alag file bhi banti hai, editor me import karne ke liye) |
| `--no-order` | narration movie ke order me nahi hai to |
| `--w-visual / --w-text / --w-ai / --w-order` | signals ka wazan (default 1 / 1 / 3 / 0.4) |
| `--back-penalty 3` | movie me peeche jaane ki saza (kam = zyada aage-peeche) |
| `--whisper-model medium` | Hindi ke liye behtar timing (thoda slow) |
| `--height 720 --fps 30 --crf 20` | output quality |

Sab options: `python -m recapcut make --help`

## Accuracy tips

1. **Subtitles zaroor do.** Movie ke `.srt` (OpenSubtitles / Subscene se, ya Whisper se khud banao). Isse dialogue matching aur `--ai` mode chalte hain.
2. **`--ai` + `--movie-title`** plot wali lines ke liye sabse accurate hai, kyunki CLIP ko "wo usse dhokha deta hai" jaisi lines samajh nahi aati.
3. Jo 2-3 clips galat lagein, unpar script me `[time]` hint lagao ya `plan.csv` edit karo.
4. Credits aur intro ko `--skip-start/--skip-end` se hatao.

## Tests

```bash
pip install pytest && python -m pytest -q tests
```

Tests ek synthetic movie banate hain (rangeen scenes + subtitles + beep wali "voice") aur check
karte hain ki sahi scene, sahi time, sahi length aur sahi rang final video me aaye.

## Credits / inspiration

Ye tool in open-source projects ke ideas par bana hai (code copy nahi kiya; kuch repos ka license
AGPL hai ya license hi nahi hai, isliye logic naye sire se likha):

- [m-abdullah-awais/movie-recap](https://github.com/m-abdullah-awais/movie-recap): CLIP shot retrieval + time-proximity scoring + clip length = narration line length
- [zcbacxc/movie-narrator](https://github.com/zcbacxc/movie-narrator): scene detect → match clips → align audio pipeline
- [zenstory-ai/video-recap-skills](https://github.com/worldwonderer/video-recap-skills): Hindi narration with edge-tts, plan-then-cut approach
- [keithhb33/AI-Movie-Shorts](https://github.com/keithhb33/AI-Movie-Shorts): subtitles se clip plan
- [linyqh/NarratoAI](https://github.com/linyqh/NarratoAI): one-click commentary workflow

## Dhyan rakhein

Movie clips copyright wale hote hain. YouTube par Content ID claim / strike aa sakti hai.
Clips chhoti rakho, apni commentary zyada rakho, aur platform ki policy check karo.
