# FlexVAE — tutorijal kroz popravke

Ovaj dokument prolazi kroz sve popravke urađene u PR #1 (kritični bagovi) i PR #2
(kvalitet treninga), ali kao lekcije: za svaku je opisano **šta je bilo pokvareno**,
**kako se to manifestovalo**, **zašto popravka izgleda baš tako** i **koja je opšta
lekcija** koja se prenosi na bilo koji budući ML projekat.

Redosled nije redosled otkrivanja, nego redosled po tome koliko je koja stvar
uticala na krajnji rezultat.

---

## 1. Skala latenta pri uzorkovanju (konstanta 0.18215)

### Šta je bilo pokvareno

Enkoder na kraju `forward` prolaza množi latent konstantom:

```python
x = self.reparametrize(self.mu, self.log_var, noise)
x *= 0.18215
```

a dekoder na početku deli istom konstantom:

```python
x /= 0.18215
```

To je konvencija preuzeta iz Stable Diffusion-a: latenti dobro istreniranog VAE-a
imaju standardnu devijaciju ~1/0.18215 ≈ 5.5, pa se skaliraju da bi difuzioni model
radio nad podacima jedinične varijanse. Za rekonstrukciju (enkoder → dekoder) sve je
konzistentno jer se množenje i deljenje poništavaju.

Problem je bio u `sample()`/`sample2()`, koji uzorkuju iz priora:

```python
noise = torch.randn((n, 4, lat, lat))   # std = 1
x = self.decoder(noise)                  # dekoder odmah deli sa 0.18215
```

Dekoder podeli šum sa 0.18215 i njegovo "telo" vidi latent sa **std ≈ 5.5** — pet i
po puta veći od svega što je ikada video tokom treninga. Generisane slike su morale
biti loše čak i da je sve ostalo bilo savršeno istrenirano.

### Popravka

```python
x = self.decoder(noise * 0.18215)
```

Šum se prvo dovede u istu skalu u kojoj enkoder isporučuje latente, pa deljenje u
dekoderu vrati std na ~1.

### Lekcija

Kad god u pipeline-u postoji konstanta skaliranja, nacrtaj tok podataka za **svaku
putanju posebno**: trening, rekonstrukcija, uzorkovanje, eksport latenata. Putanja
koja se najređe testira (ovde: uzorkovanje iz priora) je ona gde skala najlakše
"procuri". Neuralne mreže su izuzetno osetljive na ulaze van raspodele na kojoj su
trenirane — faktor 5.5 u std je katastrofa, a ne sitnica.

---

## 2. Multi-GPU trening: izgubljeni gradijenti

### Šta je bilo pokvareno

Trening pravi kopiju modela po GPU-u i batch deli srazmerno memoriji svakog GPU-a
(to je dobra ideja i zadržana je!). Svaka kopija obradi svoj deo batcha u svom
thread-u, izlazi se prebace na `cuda:0`, konkateniraju, i računa se jedan loss:

```python
predicted_image = torch.cat(sequence, dim=0)
loss = models[0].loss_function(images, predicted_image, ...)
loss.backward()
optimizer.step()          # optimizer zna SAMO za models[0].parameters()
```

`loss.backward()` korektno propagira gradijente kroz ceo graf — ali autograd
gradijente upisuje u parametre **one kopije** kroz koju je koji deo batcha prošao.
Gradijenti od chunk-ova 1..n završe u `models[1..n]` i tamo ostanu, jer optimizer
ažurira samo `models[0]`. Na početku sledećeg batcha kopije se pregaze sa
`load_state_dict(models[0].state_dict())` i njihovi gradijenti se bace.

**Posledica:** model je efektivno učio samo iz dela batcha koji je obradio GPU 0.
Sa dva GPU-a od 24 GB i 8 GB to je 75% podataka; sa jednakim GPU-ovima 50%.

### Popravka

Ključno zapažanje: sve kopije drže **identične parametre** na početku batcha, a
loss se računa jednom nad celim konkateniranim batchom. Za funkciju
L(θ) = f(chunk₀; θ) + f(chunk₁; θ) + ... važi ∂L/∂θ = Σᵢ ∂fᵢ/∂θ — tačan gradijent
celog batcha je **zbir** gradijenata po kopijama:

```python
loss.backward()
for i in range(1, num_gpu):
    for param_main, param_copy in zip(models[0].parameters(), models[i].parameters()):
        if param_copy.grad is not None:
            param_main.grad.add_(param_copy.grad.to(param_main.device))
            param_copy.grad = None
```

Ovo je, u malom, upravo ono što radi `DistributedDataParallel` (all-reduce
gradijenata posle backward-a), samo što DDP usrednjava preko rank-ova pa
pretpostavlja jednake batch-eve, dok ovde zbir radi tačno i sa nejednakim
chunk-ovima.

### Kako je verifikovano

CPU smoke test: dve kopije modela sa sinhronizovanim težinama obrade nejednake
delove batcha (3+1), zbir njihovih gradijenata se uporedi sa gradijentom jednog
modela koji obradi ceo batch. Maksimalna razlika: 6·10⁻⁵ (šum float32 akumulacije —
sabiranje istih brojeva drugim redosledom daje sitno drugačiji rezultat).

### Lekcija

Autograd upisuje gradijent u `.grad` polja **onih tenzora kroz koje je graf
prošao** — ne "u model" apstraktno. Kod svake ručne data-parallel šeme postavi
sebi pitanje: *gde tačno završi gradijent svakog dela batcha, i ko ga primeni?*
I drugo: ovakve stvari se mogu numerički verifikovati na CPU-u malim testom
ekvivalencije — to je najjači alat za proveru korektnosti gradijenata.

---

## 3. KLD nad statistikama celog batcha

### Šta je bilo pokvareno

Povezano sa prethodnim: `loss_function` je koristila `self.encoder.mu` i
`self.encoder.log_var` — atribute koje enkoder upiše tokom `forward`-a. Ali to su
statistike **samo onog chunk-a koji je prošao kroz `models[0]`**. Rekonstrukcioni
loss je pokrivao ceo batch, a KLD i sparse član samo prvi chunk — dva dela loss-a
su gledala različite podatke.

### Popravka

GPU thread-ovi sada vraćaju i `mu`/`log_var` svog sub-batcha, koji se konkateniraju
kao i predikcije, a `loss_function` prima opcione parametre `mu` i `log_var`
(default su i dalje atributi enkodera, pa single-GPU poziv radi kao pre).

### Lekcija

Prenošenje međurezultata kroz atribute objekta (`self.mu = ...` u `forward`-u) je
zgodno, ali je **skriveno stanje**: čim postoji više instanci modela ili više
poziva, lako se desi da čitaš stanje pogrešne instance/poziva. Eksplicitno
prosleđivanje vrednosti kroz povratne vrednosti i argumente je dosadnije, ali se
ovakvi bagovi tada vide golim okom.

---

## 4. Enkoder koji ne vidi 25% slike (stride > kernel)

### Šta je bilo pokvareno

Downsampling slojevi enkodera bili su:

```
C2d_128_128_3_4_0   →  nn.Conv2d(128, 128, kernel_size=3, stride=4, padding=0)
```

Kernel 3, stride 4. Prozor na poziciji `i` pokriva piksele `i, i+1, i+2`, sledeći
počinje na `i+4` — piksel `i+3` **ne upada ni u jedan prozor**. Svaki četvrti red i
svaka četvrta kolona slike nikada ne doprinose latentu. Konvolucija bukvalno ne
vidi 25% ulaza, i to u pravilnoj rešetki — plafon za kvalitet rekonstrukcije i
recept za mrežaste artefakte.

### Popravka

`model_structures/VAE_encoder_v2.mstruct` menja te slojeve u kernel 4, stride 4:

```
C2d_128_128_4_4_0
```

Puna pokrivenost, bez preklapanja, iste izlazne dimenzije (256 → 64 → 16). Fajl je
opt-in (biraš ga kroz `encoder_struct` u konfiguraciji) jer kernel 4 ima drugačiji
oblik težina od kernela 3 — stari checkpointi nisu kompatibilni.

Napomena: standardni SD-VAE ovo rešava sa dva stride-2 sloja (kernel 3, stride 2,
uz asimetrični pad) umesto jednog stride-4 — blaži downsampling u dva koraka obično
daje još bolje rezultate, ali menja dubinu mreže, pa je kernel-4 varijanta
najmanja izmena koja uklanja rupe.

### Lekcija

Za svaki konvolucioni sloj proveri odnos kernela i stride-a: **ako je
`stride > kernel_size`, delovi ulaza se garantovano preskaču.** Ovo ne prijavljuje
nijedan alat — dimenzije izlaza ispadnu "lepe" (256/4 = 64) i sve radi, samo lošije
nego što bi moglo.

---

## 5. Težina KLD člana (zašto su uzorci bili loši, a rekonstrukcije pristojne)

### Pozadina: čemu služi KLD u VAE

VAE loss ima dva takmičarska člana:

- **rekonstrukcija** (MSE/SSIM): "dekodiraj tačno ono što je ušlo",
- **KLD**: "neka posterior q(z|x) = N(μ, σ²) liči na prior N(0, 1)".

KLD je ono što čini latentni prostor *upotrebljivim za generisanje*: bez njega
enkoder razbaca slike po latentu kako mu odgovara (postaje običan autoencoder), pa
uzorak iz N(0,1) padne u "rupu" između njih i dekoder vrati đubre. Prejak KLD pak
izaziva *posterior collapse* — latent ignoriše ulaz i sve rekonstrukcije liče na
prosek dataseta.

### Šta je bilo u kodu

```python
kld_weight=(epoch / 100) - int(epoch / 100)  # mrtav kod, odmah pregažen
kld_weight=1/4*0.01                           # = 0.0025, zakucano
...
kld_loss = kld_loss/(c*ld)                    # još podeljeno sa ~1024
```

uz `recons_loss = 400*F.mse_loss(...)`. Odnos rekonstrukcije i KLD-a bio je reda
10⁵–10⁶ : 1 — KLD je bio praktično dekorativan. Zato su rekonstrukcije bile OK
(model je de facto autoencoder), a uzorkovanje loše (posterior nema veze sa
priorom). U kombinaciji sa bagom #1 (skala ×5.5), uzorci nisu imali šanse.

### Popravka

Težina je izvučena u konfiguraciju (`"kld_weight": 0.0025` — default čuva staro
ponašanje), pa se sada tjunira bez diranja koda. Preporučeni eksperiment: sweep
0.0025 → 0.01 → 0.05 → 0.1 i gledaj obe krive (rekonstrukcija i KLD) + vizuelni
kvalitet uzoraka.

### Lekcija

Hiperparametri koji odlučuju o ponašanju modela ne smeju biti zakucani u kodu, a
pogotovo ne kao "mrtva" linija koja odmah pregazi prethodnu formulu — to je znak da
je eksperimentisanje ostavilo tragove koje niko više ne razume. I šire: **balans
članova loss-a proveri brojkama** (isprintaj svaki član posebno — kod to već radi
kroz TensorBoard!) umesto da veruješ konstantama.

Napredna tema kad dođeš do nje: *KLD annealing* (kreni od 0 i postepeno dižiš
težinu, ili ciklično) i *free bits* — standardni trikovi da se dobije i dobra
rekonstrukcija i dobar prior match.

---

## 6. Learning rate koji je odumro (StepLR geometrija)

### Šta je bilo

```python
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
```

Polovljenje LR-a svakih 100 epoha zvuči bezazleno, ali je geometrijski niz:
checkpoint u repou je na **epohi 1777**, što znači 0.5¹⁷ ≈ 1/131.000 početnog LR-a
— reda 10⁻⁹. Od neke epohe koraci optimizacije su praktično nula i "trening" samo
troši struju. Simptom: "treniram još stotine epoha i ništa se ne menja".

### Šta je popravljeno (i šta nije)

Popravljena su dva mehanička problema: `scheduler2` se više ne poziva kad je
`use_scheduler` isključen (ranije `NameError`), a `ReduceLROnPlateau` sada gleda
prosečan loss epohe umesto loss-a poslednjeg batcha (pojedinačan batch je previše
šuman signal za detekciju platoa).

Sam izbor scheduler-a je odluka za tebe: za ovako duge trening-ove razumnije je
cosine sa warmup-om, ili konstantan LR + `ReduceLROnPlateau`. Ako ostaješ na
StepLR-u, izračunaj unapred koliki će LR biti na kraju planiranog broja epoha.

### Lekcija

Za svaki scheduler napiši na papiru vrednost LR-a na epohama 100, 500, 1000. I
loguj LR (kod ga sada prikazuje u progress baru direktno iz optimizatora) — mrtav
LR je jedan od najčešćih "misterioznih" razloga zašto model stagnira.

---

## 7. Bool argumenti iz komandne linije: `bool('False') == True`

### Šta je bilo

Argumenti se generišu iz config fajla, sa tipom izvedenim iz vrednosti:

```python
parser.add_argument(f'-{key}', nargs=1, type=type(value), ...)
```

Za boolean ključeve to daje `type=bool`, a `bool` nad stringom vraća `True` za
**svaki neprazan string** — uključujući `'False'`. Dakle `-reinit_optim False` je
postavljao `True`. Zlokobno je što `-nesto True` radi "ispravno", pa bag ostaje
neprimećen dok jednom ne pokušaš da nešto isključiš.

### Popravka

```python
def str2bool(value):
    if value.lower() in ('true', 't', 'yes', 'y', '1'):  return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):  return False
    raise argparse.ArgumentTypeError(...)

arg_type = str2bool if isinstance(value, bool) else type(value)
```

### Lekcija

`argparse` sa `type=bool` je poznata Python zamka: `type` je samo funkcija
`str -> vrednost`, a `bool()` nije parser nego provera praznine. Uvek testiraj CLI
sa vrednošću koja *isključuje* opciju, ne samo sa onom koja je uključuje.

---

## 8. Rušenja kod isključenih opcija (SSIM, scheduler, `-d` flag)

Tri baga istog porekla:

- `loss_function(..., ssim_metrics=False)` → `UnboundLocalError`, jer se
  `ssim_loss` definisao samo u `if` grani, a koristio bezuslovno u return-u;
- `scheduler2.step()` se pozivao i kad `use_scheduler == False`, a objekat se tada
  uopšte ne kreira → `NameError`;
- `-d` (decode) režim nije bio u listi `exclusive_flags`, pa je provera "tačno
  jedan flag" uvek padala za `-d` → režim neupotrebljiv, iako je u README-u.

### Lekcija

Svaka opcija koja grana kod ima **dve** putanje, a testira se obično samo jedna
(ona koju sam koristiš). Minimalan higijenski standard: za svaki `if opcija:` koji
kreira objekat ili promenljivu, proveri šta se dešava sa svim kasnijim upotrebama
kad je opcija isključena. Ovo je i najjači argument za makar minimalne testove —
sva tri baga bi uhvatio test koji samo *pozove* kod sa isključenim opcijama.

---

## 9. Augmentacija na pogrešnom mestu

### Šta je bilo

`RandomHorizontalFlip(p=0.5)` je stajao u `load_image` — funkciji koja se koristi
za **inferencu, enkodovanje i interpolaciju** — a trening transformacije
(`get_data`) nisu imale nikakvu augmentaciju.

Posledice: `encode_from_folder` je nasumično flipovao pola slika dok pravi latent
dataset (pa latent sačuvan pod imenom slike ne odgovara toj slici!), inferenca je
bila nedeterministička, a trening nije dobijao benefit augmentacije.

### Popravka

Flip premešten u `get_data` (trening); `load_image` je sada determinističan.

### Lekcija

Zlatno pravilo: **augmentacija ide isključivo u trening pipeline.** Inferenca i
evaluacija moraju biti determinističke — inače ne možeš porediti rezultate, a
izvedeni artefakti (latenti, metrike) postaju nasumično iskvareni. Kad deliš
transformacije između treninga i inference, budi eksplicitan koja verzija ide gde.

---

## 10. SSIM bez fiksnog `data_range`

`SSIM(pred, target)` bez `data_range` pušta torchmetrics da opseg proceni iz
min/max **svakog batcha**. Pošto izlaz dekodera nije ograničen (nema tanh/clamp),
opseg varira od batcha do batcha, pa ista stvarna sličnost daje različit loss —
šum u signalu za učenje. Slike su normalizovane na [-1, 1], pa je tačan opseg 2:

```python
ssim_loss = SSIM(reconstructed_image, input_image, data_range=2.0)
```

**Lekcija:** metrike koje zavise od opsega podataka (SSIM, PSNR) uvek dobijaju
eksplicitan `data_range`. Podrazumevane vrednosti biblioteka su pogodne za brzu
probu, ne za loss funkciju.

---

## 11. EMA jednom po epohi

`AveragedModel` sa decay-em 0.999 je dizajniran da se ažurira **posle svakog
optimizer koraka**: efektivno usrednjava poslednjih ~1000 koraka. Ažuriranje jednom
po epohi sa istim decay-em usrednjava poslednjih ~1000 epoha — tj. praktično ništa
korisno. Update je premešten odmah iza `optimizer.step()`.

**Lekcija:** decay konstanta i frekvencija ažuriranja su jedan par — ne mogu se
birati nezavisno. Kad preuzimaš tehniku (EMA, warmup, annealing), proveri *na koju
vremensku osu* se njeni parametri odnose (korak vs epoha).

Napomena: EMA model se trenutno ne koristi pri uzorkovanju niti se snima u
checkpoint — to je smisleno sledeće poboljšanje ako uključiš `useEMA`.

---

## 12. Dupli dropout

U `VAE_Encoder.forward` isti `if` blok je stajao dva puta zaredom:

```python
if (type(module) == Dropout2d and self.dropout_early==True):
    x = module(x)
if (type(module) == Dropout2d and self.dropout_early==True):   # copy-paste
    x = module(x)
```

Dropout sa stopom p, primenjen dvaput, daje efektivnu stopu 1-(1-p)² — za p=0.2 to
je 0.36, skoro duplo jača regularizacija nego što piše u kodu.

**Lekcija:** copy-paste u `forward` metodama je posebno opasan jer ne pravi grešku
nego *tiho menja model*. Ovakve stvari se hvataju čitanjem diff-a pre commit-a.

---

## 13. Ignorisan `weight_decay`

Config je imao `"weight_decay": 0.01`, ali se `optim.AdamW(...)` kreirao bez tog
argumenta — važio je default biblioteke. Vrednosti se trenutno poklapaju, pa bag
nije menjao ponašanje, ali bi svako podešavanje kroz config bilo tiho ignorisano.

**Lekcija:** za svaki ključ u konfiguraciji mora postojati mesto u kodu koje ga
čita. Povremeno pretraži projekat za svaki config ključ — ključ koji se nigde ne
čita je ili mrtav, ili (gore) misliš da nešto podešavaš a ne podešavaš.

---

## 14. Kako je sve ovo testirano bez GPU-a

Za projekat učenja ovo je možda najkorisnija lekcija: skoro sve gore navedeno je
verifikovano **na CPU-u, malim smoke testovima**, iako je kod pisan za CUDA:

1. **Ekvivalencija gradijenata** (bag #2): dve kopije modela na CPU-u simuliraju
   dva GPU-a; zbir njihovih gradijenata se poredi sa referentnim gradijentom celog
   batcha. Ako se poklapaju do float šuma — logika je tačna, nezavisno od uređaja.
2. **Skala uzorkovanja** (bag #1): čista aritmetika nad tenzorima —
   `(noise * 0.18215) / 0.18215` ima std ~1, `noise / 0.18215` ima std ~5.5.
3. **Isključene opcije** (bag #8): pozovi `loss_function(ssim_metrics=False)` i
   proveri da vrati konačan broj.
4. **CLI parsiranje** (bag #7): `str2bool('False') is False`, i to je ceo test.
5. **Nova arhitektura** (bag #4): sagradi model iz `VAE_encoder_v2.mstruct`,
   proturi batch, proveri dimenzije.

Nijedan od ovih testova ne zahteva GPU, dataset niti trening — a zajedno hvataju
većinu klasa bagova iz ovog dokumenta. Vredi ih pretvoriti u prave `pytest`
testove u `tests/` folderu (koji već postoji, ali je prazan).

---

## Kuda dalje

Redosled eksperimenata koji ima smisla posle ovih popravki:

1. **Retrening od nule** sa `VAE_encoder_v2` (bez rupa u pokrivenosti) i
   popravljenim uzorkovanjem — tek to daje pravu sliku koliko model može.
2. **Sweep `kld_weight`** (0.0025 → 0.01 → 0.05 → 0.1): traži se tačka gde uzorci
   iz priora postaju smisleni, a rekonstrukcija još nije primetno degradirala.
   Zatim eventualno KLD annealing.
3. **Razuman LR režim**: cosine sa warmup-om ili konstantan LR + plateau; proveri
   efektivni LR pri svakom resume-u.
4. Ako rekonstrukcije i dalje nisu dovoljno dobre: **f8 umesto f16** (latent
   32×32×4 za sliku 256×256) — kompresija 48:1 umesto 192:1.
5. Za generisanje: difuzija nad latentima (već postoji `decode_from_diffusion`
   putanja) je standardan i najisplativiji put — oblik priora VAE-a tada gotovo ne
   igra ulogu, bitni su kvalitet rekonstrukcije i glatkoća latentnog prostora.
