# FlexVAE — step-by-step vodič za sledeći trening

Ovaj dokument je praktični plan rada: šta uraditi, kojim redom, i **zašto**
svaki korak postoji. Namenjen je učenju — ne samo da „prođe“, nego da razumeš
šta meriš.

Ako želiš teoriju iza ranijih bagova, čitaj i
[TUTORIAL.md](TUTORIAL.md) (šta je bilo pokvareno i koje lekcije).

---

## Preduslovi

1. Spoji (ili checkout-uj) grane sa popravkama:
   - PR #1 kritični bagovi (multi-GPU gradijenti, CLI, SSIM/scheduler)
   - PR #2 kvalitet treninga (skala uzorkovanja, encoder_v2, kld_weight…)
   - ova grana: scheduler / EMA / annealing / AMP / testovi
2. Instaliraj zavisnosti:

```bash
pip install -r requirements.txt
```

3. Proveri da testovi prolaze (ne zahtevaju GPU):

```bash
PYTHONPATH=src pytest -q
```

Ako nešto padne ovde, ne kreći u dug trening — prvo popravi okruženje.

---

## Korak 1 — Novi run od nule (ne resume starog checkpointa)

**Zašto:** stari checkpointi su trenirani sa izgubljenim multi-GPU gradijentima,
pogrešnom skalom uzorkovanja i (ako koristiš v2 enkoder) drugačijim oblikom
težina. Resume ih ne „popravlja“ — samo nastavlja lošu putanju.

U `configs/config.cfg` postavi:

```json
"resume": false,
"run_name": "vae256_v2_kld005_cosine",
"encoder_struct": "VAE_encoder_v2",
"decoder_struct": "VAE_decoder",
"image_size": 256,
"lat_size": 16,
"latent_conversion_disable": true,
"dataset_path": "/putanja/do/tvog/dataseta"
```

`VAE_encoder_v2` koristi kernel-4 stride-4 downsampling (nema rupa u pokrivenosti
piksela). Detalj: [TUTORIAL.md §4](TUTORIAL.md).

---

## Korak 2 — Uključi razuman LR režim

**Zašto:** stari `StepLR(step=100, gamma=0.5)` na dugom treningu ubija learning
rate (geometrijski niz). Cosine spušta LR glatko do kraja plana.

```json
"use_scheduler": true,
"scheduler_type": "cosine",
"cosine_t_max": 0,
"cosine_eta_min": 0.0,
"lr": 0.0004,
"ReduceLROnPlateau": false
```

- `cosine_t_max: 0` znači „koristi `epochs`“ — LR kriva prati tvoj plan treninga.
- Ako želiš stari StepLR: `"scheduler_type": "step"`.

U progress baru prati `learn_rate`. Ako posle 200 epoha vidiš vrednosti reda
`1e-8` bez namere — scheduler ti je opet „ugasio“ učenje.

---

## Korak 3 — KLD težina + (opciono) annealing

**Zašto:** bez dovoljnog KLD-a model postaje autoencoder (dobre rekonstrukcije,
loši uzorci iz priora). Sa prejakim KLD-om kolabira posterior (loše rekonstrukcije).

Početna preporuka za prvi pravi run:

```json
"kld_weight": 0.05,
"kld_anneal": "linear",
"kld_anneal_epochs": 100
```

Šta se dešava:

- epohe 0–100: KLD težina raste od 0 → 0.05 (model prvo uči da rekonstruiše)
- posle 100: ostaje 0.05

U TensorBoard-u ćeš videti krivu `kld_weight` i član `KLD`.

### Mini-eksperiment (najvažniji za učenje)

Uradi 4 kratka run-a (~50 epoha), menjajući samo `kld_weight`
(sa `kld_anneal: "none"` da poređenje bude čisto):

| run | kld_weight | šta očekuješ |
|-----|------------|--------------|
| A | 0.01 | bolja rekonstrukcija, slabiji uzorci |
| B | 0.05 | balans |
| C | 0.1 | jači prior match |
| D | 0.2 | rizik posterior collapse |

Za svaki run uporedi:

1. TensorBoard: `Reconstruction_Loss` vs `KLD`
2. `results/<run_name>/` — vizuelna rekonstrukcija
3. `samples/<run_name>/` — uzorci iz priora

**Cilj učenja:** da sam vidiš trade-off, ne da „pogodiš magični broj“.

Alternativa annealing-u: `"kld_anneal": "cyclical"` + `"kld_anneal_cycle": 50`
(ponavlja ramp 0→target unutar ciklusa). Korisno ako linear annealing „zaključa“
model prerano.

---

## Korak 4 — DataLoader: ne gladuj GPU

```json
"num_workers": 4,
"pin_memory": true
```

**Zašto:** bez worker-a CPU serijski učitava slike dok GPU čeka. `pin_memory`
ubrzava host→GPU transfer. Ako dobiješ greške oko shared memory / fork-a na
Windows-u, spusti `num_workers` na 0.

---

## Korak 5 — EMA za stabilnije uzorke

```json
"useEMA": true,
"ema_decay": 0.999,
"sample_with_ema": true
```

**Zašto:** EMA usrednjava težine kroz poslednje ~1000 optimizer koraka. Često daje
stabilnije generisane slike od „sirovog“ modela. Checkpoint sada čuva i
`ema_state_dict`, pa resume radi.

`sample_with_ema: true` znači da se per-epoch uzorci u `samples/` crte iz EMA
težina.

---

## Korak 6 — (Opciono) ubrzanja: AMP i torch.compile

```json
"use_amp": true,
"torch_compile": false
```

- **AMP** (`use_amp`): mixed precision — obično 1.5–2× brže na modernim NVIDIA
  karticama, uz `GradScaler` koji čuva stabilnost. Probaj tek kad baseline radi.
- **torch.compile**: opt-in; najkorisnije na jednom GPU-u. Prvi epoch je sporiji
  (kompajliranje), zatim ubrzanje. Sa custom multi-GPU thread-ovima može biti
  kapriciozno — zato default `false`.

---

## Korak 7 — Pokreni trening

```bash
python3 ./train.py -t
```

Ili override bez editovanja fajla:

```bash
python3 ./train.py -t \
  -run_name vae256_v2_kld005_cosine \
  -resume False \
  -encoder_struct VAE_encoder_v2 \
  -kld_weight 0.05 \
  -kld_anneal linear \
  -scheduler_type cosine \
  -useEMA True \
  -wo
```

`-wo` upisuje izmene nazad u `config.cfg` (korisno da sačuvaš tačan setup run-a).

Prati:

```bash
tensorboard --logdir runs
```

Gledaj: `Total_loss`, `Reconstruction_Loss`, `KLD`, `kld_weight`, learning rate
u progress baru.

---

## Korak 8 — Kako oceniti da li je „dobro“

Posle ~50–100 epoha pitaj se redom:

1. **Rekonstrukcija** (`results/`) — da li se prepoznaje sadržaj ulaza?
2. **Uzorci** (`samples/`) — da li liče na dataset ili su šum/blob?
3. **KLD kriva** — da li uopšte raste / stabilizuje se, ili je numerički mrtva?
4. **LR** — da li je još u aktivnom opsegu?

Ako je rekonstrukcija dobra a uzorci loši → povećaj `kld_weight` ili produži
annealing.
Ako su i rekonstrukcije i uzorci loši → proveri dataset path, batch size, da li
si na `encoder_v2`, da li radi GPU trening.
Ako rekonstrukcija stagnira sa jakim KLD-om → smanji težinu ili koristi linear
annealing.

---

## Korak 9 — Šta posle dobrog VAE-a

Kad rekonstrukcije budu prihvatljive:

1. **Encode dataset** u latente: `python3 ./train.py -e`
2. Treniraj difuzioni model nad tim latentima (to je Stable Diffusion recept).
   Oblik priora VAE-a tada manje biti — bitni su kvalitet rekonstrukcije i
   glatkoća latentnog prostora.
3. Ako detalji i dalje nestaju: pređi na manju kompresiju (f8, latent 32×32 za
   256 sliku) — to je arhitekturni sledeći korak, veći od hiperparametara.

---

## Brzi cheat-sheet novih config ključeva

| ključ | smisao | razumna početna |
|-------|--------|-----------------|
| `scheduler_type` | `cosine` / `step` / `none` | `cosine` |
| `kld_weight` | ciljna težina KLD člana | `0.05` (sweep!) |
| `kld_anneal` | `none` / `linear` / `cyclical` | `linear` |
| `kld_anneal_epochs` | trajanje linear ramp-a | `100` |
| `kld_anneal_cycle` | dužina cyclical ciklusa | `50` |
| `num_workers` | DataLoader worker-i | `4` |
| `pin_memory` | brži CPU→GPU copy | `true` |
| `useEMA` | EMA težine | `true` za sample kvalitet |
| `ema_decay` | EMA α | `0.999` |
| `sample_with_ema` | uzorci iz EMA | `true` |
| `use_amp` | mixed precision | `true` kad baseline radi |
| `torch_compile` | PyTorch 2 compile | `false` dok ne meriš |

---

## Checklist pre dugog run-a

- [ ] `pytest -q` prolazi
- [ ] `resume: false` za novi eksperiment
- [ ] `encoder_struct: VAE_encoder_v2`
- [ ] `scheduler_type: cosine`
- [ ] odabran `kld_weight` (ili plan sweep-a)
- [ ] `dataset_path` pokazuje na pravi folder (`ImageFolder` struktura)
- [ ] TensorBoard spreman
- [ ] znaš gde gledaš `results/` i `samples/`

Kad završiš prvi uspešan run, vrati se na [TUTORIAL.md](TUTORIAL.md) i poveži
simptome koje si video sa lekcijama — to je ceo smisao projekta kao učenja.
