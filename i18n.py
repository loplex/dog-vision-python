"""The languages dog_vision's window speaks.

As in gettext, a text is looked up by its English wording, and a text missing from
a language's catalogue stays English; that is how source citations pass through
unchanged. Adding a language is adding an entry to LANGUAGES.
"""

import dataclasses
import locale
import os
import sys

CZECH = {
    # Window
    "Dog vision": "Psí vidění",
    "Species": "Druh",
    "Selected species": "Vybraný druh",
    "Adaptation to scene [%]": "Adaptace na scénu [%]",
    "Simulation strength [%]": "Síla simulace [%]",
    "Colour saturation": "Sytost barev",
    "Fixed by the projection": "Daná projekcí",
    "Matched to discrimination (RNL)": "Podle rozlišování (RNL)",
    "Acuity": "Ostrost",
    "Blur to the species' acuity": "Rozostřit na ostrost druhu",
    "Image spans [degrees]": "Šířka záběru [°]",
    "View": "Zobrazení",
    "Side by side (m)": "Vedle sebe (m)",
    "Left image": "Levý obraz",
    "Map of differences (d)": "Mapa rozdílů (d)",
    "Reset (r)": "Obnovit (r)",
    "Save snapshot (s)": "Uložit snímek (s)",
    "Saved {name}": "Uloženo: {name}",
    "No frame yet": "Zatím žádný snímek",
    "Language": "Jazyk",
    "Source": "Zdroj",
    "Camera {index}": "Kamera {index}",
    "Open file… (o)": "Otevřít soubor… (o)",
    "Camera": "Kamera",
    "Convert file": "Převést soubor",
    "Open a photo or a video": "Otevřít fotku nebo video",
    "Photos and videos": "Fotky a videa",
    "All files": "Všechny soubory",
    "Cannot open {name} as a photo or a video": "{name} nejde otevřít jako fotka ani jako video",
    "Cannot open camera {index}": "Kameru {index} nejde otevřít",
    "Converting: {share}": "Převádím: {share}",
    "Wrote {name}": "Zapsáno: {name}",
    "Wrote {name}: {description}": "Zapsáno: {name}, {description}",
    "Conversion failed: {error}": "Převod selhal: {error}",
    "{format} ({encoder}), with the original sound": "{format} ({encoder}), s původním zvukem",
    "{format} ({encoder}); the original has no sound": "{format} ({encoder}); originál nemá zvuk",
    "{format} ({encoder}), without sound: ffmpeg is not installed": "{format} ({encoder}), bez zvuku: ffmpeg není nainstalovaný",
    "{format} ({encoder})": "{format} ({encoder})",
    # Caption
    "original": "originál",
    "left: {left}    right: {right}": "vlevo: {left}    vpravo: {right}",
    "    red: noticeably different ({share} of pixels)": "    červeně: znatelně odlišné ({share} pixelů)",
    # Species facts
    "{0}%": "{0} %",
    "{0}%–{1}%": "{0}–{1} %",
    "Colour vision": "Barevné vidění",
    "{kind}, 1 cone type": "{kind}, 1 typ čípků",
    "{kind}, {n} cone types": "{kind}, {n} typy čípků",
    "Cone peaks": "Maxima čípků",
    "Peaks from": "Zdroj maxim",
    "S cones": "Čípky S",
    "{share} of cones ({source})": "{share} čípků ({source})",
    "L : M cones": "Poměr L : M",
    "{ratio} : 1 (assumed)": "{ratio} : 1 (předpoklad)",
    "Neutral point": "Neutrální bod",
    "RNL scale": "Škála RNL",
    "nothing to scale": "není co škálovat",
    "{gains} of fixed": "{gains} vůči projekci",
    " and ": " a ",
    "assumed": "předpoklad",
    "not found measured; left sharp": "měření nenalezeno; ponecháno ostré",
    "{value} c/deg": "{value} c/°",
    "{across} c/deg side by side, {up} one above another": "{across} c/° vodorovně, {up} svisle",
    # Notes attached to a source in dog_vision's tables
    "L shifted 10 nm, see text": "L posunutý o 10 nm, viz text",
    "M shifted 10 nm, see text": "M posunutý o 10 nm, viz text",
    "Jacobs et al. 1996, S assumed": "Jacobs et al. 1996, S předpokládán",
    "Sumita et al. 2013, 11.7 to 14": "Sumita et al. 2013, 11,7 až 14",
    "Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8": "Jacobs, Birch & Blakeslee 1982, 1,8 až 3,8",
    "Nature Neuroscience 2024, about 60": "Nature Neuroscience 2024, asi 60",
    "Hanke & Dehnhardt 2009, in air": "Hanke & Dehnhardt 2009, na vzduchu",
    "Herman et al. 1975, 8.2 arcmin stripes": "Herman et al. 1975, pruhy 8,2′",
    # Descriptions shown while the pointer rests on a label
    (
        "How many kinds of cone the species has, and so how many independent colour signals its eye sends."
        "\n\nA trichromat, like us, has three. A camera records nothing that would merge for it, so its image"
        " stays as it is unless Colour saturation is matched to discrimination."
        "\n\nA dichromat has two. Colours that differ only along one direction look the same to it, and the"
        " simulation shows them on the blue–yellow axis."
        "\n\nA cone monochromat has one, and sees shades of grey."
    ): (
        "Kolik druhů čípků živočich má, a tedy kolik nezávislých barevných signálů jeho oko vysílá."
        "\n\nTrichromat jako my má tři. Kamera nezaznamená nic, co by mu splývalo, takže jeho obraz zůstane"
        " beze změny, pokud Sytost barev není nastavená Podle rozlišování."
        "\n\nDichromat má dva. Barvy, které se liší jen v jednom směru, pro něj vypadají stejně, a simulace je"
        " ukáže na ose modrá–žlutá."
        "\n\nČípkový monochromat má jeden a vidí odstíny šedi."
    ),
    (
        "The wavelength each kind of cone is most sensitive to: S for short, M for middle and L for long"
        " wavelengths. A dichromat's longer cone is listed as L whatever its source calls it."
        "\n\nEach cone is modelled from its peak alone, with the pigment template of Govardovskii et al."
        " (2000). Its sensitivity is about 100 nm wide at half height, which is why a shift of a few"
        " nanometres between species changes little."
        "\n\nA human's cones peak at 420.7, 530.3 and 558.9 nm."
    ): (
        "Vlnová délka, na kterou je každý druh čípku nejcitlivější: S pro krátké, M pro střední a L pro"
        " dlouhé vlnové délky. Delší čípek dichromata je uveden jako L, ať ho zdroj nazývá jakkoli."
        "\n\nKaždý čípek je modelován jen z tohoto maxima, šablonou pigmentu, kterou popsali Govardovskii et"
        " al. (2000). Jeho citlivost je v polovině výšky široká asi 100 nm, a proto posun o pár nanometrů"
        " mezi druhy změní málo."
        "\n\nLidské čípky mají maxima 420,7; 530,3 a 558,9 nm."
    ),
    (
        "The study the cone peaks are taken from; the README lists every source in full under References."
        "\n\nA note after the citation marks a value that is not simply measured, such as a cone shifted on"
        " purpose or one assumed."
    ): (
        "Studie, ze které jsou maxima čípků převzata; README uvádí všechny zdroje celé v části References."
        "\n\nPoznámka za citací označuje hodnotu, která není prostě změřená, třeba záměrně posunutý čípek nebo"
        " předpokládanou hodnotu."
    ),
    (
        "The share of all cones that are S cones, with its source. Where the source reports a range across"
        " the retina, the middle of it is used."
        "\n\nOnly the RNL colour saturation uses it: the fewer cones of a kind, the noisier their signal, and"
        " the larger a colour difference along that cone's axis must be to be noticed."
        "\n\n\"Assumed\" means no measurement was found, and 10 % is used, the middle of what the measured"
        " species span."
    ): (
        "Podíl čípků S mezi všemi čípky, se zdrojem. Kde zdroj uvádí rozpětí napříč sítnicí, použije se jeho"
        " střed."
        "\n\nPoužívá ho jen sytost barev Podle rozlišování (RNL): čím méně čípků daného druhu, tím zašuměnější"
        " je jejich signál a tím větší musí být barevný rozdíl podél jejich osy, aby byl postřehnutelný."
        "\n\n„Předpoklad“ znamená, že měření nebylo nalezeno a použije se 10 %, střed rozpětí změřených druhů."
    ),
    (
        "How many L cones a trichromat has for every M cone. Only the RNL colour saturation uses it."
        "\n\nHumans with normal colour vision range from about twice as many L as M cones to the reverse"
        " (Roorda & Williams 1999), so every trichromat is given 1 : 1, and it is marked as assumed."
    ): (
        "Kolik čípků L má trichromat na každý čípek M. Používá ho jen sytost barev Podle rozlišování (RNL)."
        "\n\nLidé s normálním barevným viděním mají od zhruba dvojnásobku čípků L oproti M až po opačný poměr"
        " (Roorda & Williams 1999), proto má každý trichromat 1 : 1 a hodnota je označená jako předpoklad."
    ),
    (
        "The wavelength of pure spectral light that a dichromat cannot tell from white. Shorter wavelengths"
        " look bluish to it, longer ones yellowish."
        "\n\nThe value is the model's, computed from the cone peaks. For the dog it matches the measured"
        " neutral point of about 480 nm (Neitz, Geist & Jacobs 1989). For a human deuteranope it falls well"
        " short of the measured 505 nm, probably because the model ignores the lens and the macular pigment."
    ): (
        "Vlnová délka čistého spektrálního světla, které dichromat nerozezná od bílé. Kratší vlnové délky mu"
        " připadají namodralé, delší nažloutlé."
        "\n\nHodnota pochází z modelu, spočítaná z maxim čípků. U psa odpovídá změřenému neutrálnímu bodu kolem"
        " 480 nm (Neitz, Geist & Jacobs 1989). U lidského deuteranopa vychází výrazně pod změřenými 505 nm,"
        " nejspíš proto, že model nezahrnuje čočku ani makulární pigment."
    ),
    (
        "How much the RNL colour saturation changes saturation compared with the fixed one. Below 1 the"
        " species tells colours apart worse than we do, above 1 better."
        "\n\nIt comes from the receptor noise limited model of Vorobyev & Osorio (1998), which counts"
        " just-noticeable differences from each cone's noise, set by the share of cones of its kind. The"
        " output is rescaled until a human looking at it counts as many differences as the animal would. A"
        " trichromat has two colour axes and so two factors; a monochromat has none."
        "\n\nIt assumes the animal's cones are as noisy as ours, which is measured for almost no mammal."
    ): (
        "O kolik sytost barev Podle rozlišování (RNL) mění sytost oproti sytosti dané projekcí. Pod 1"
        " rozlišuje druh barvy hůř než my, nad 1 lépe."
        "\n\nVychází z modelu šumem omezených receptorů (receptor noise limited, Vorobyev & Osorio 1998), který"
        " počítá právě postřehnutelné rozdíly ze šumu každého čípku, daného podílem čípků jeho druhu. Výstup"
        " se přeškáluje, dokud člověk, který se na něj dívá, nenapočítá tolik rozdílů, kolik by jich"
        " napočítal živočich. Trichromat má dvě barevné osy, a tedy dva faktory; monochromat žádný."
        "\n\nPředpokládá, že čípky živočicha jsou stejně zašuměné jako naše, což je změřené skoro u žádného"
        " savce."
    ),
    (
        "The finest stripes the species resolves, in cycles per degree: pairs of one dark and one light"
        " stripe within one degree of view. A human resolves about 72."
        "\n\nWith Blur to the species' acuity checked, detail finer than that is removed, as AcuityView does"
        " (Caves & Johnsen 2018)."
        "\n\nCattle resolve detail side by side better than one above another, since their pupil is a"
        " horizontal oval. A species without a measurement is left sharp."
    ): (
        "Nejjemnější pruhy, které druh rozliší, v cyklech na stupeň: dvojice jednoho tmavého a jednoho"
        " světlého pruhu v jednom stupni zorného pole. Člověk rozliší asi 72."
        "\n\nPři zaškrtnutém Rozostřit na ostrost druhu se odstraní detail jemnější než tato hranice, stejně"
        " jako to dělá AcuityView (Caves & Johnsen 2018)."
        "\n\nSkot rozliší detail vedle sebe lépe než nad sebou, protože má zornici ve tvaru vodorovného oválu."
        " Druh bez měření zůstane ostrý."
    ),
    (
        "The animal to simulate, with its kind of colour vision in brackets."
        "\n\nMost dichromatic mammals look much alike: they share the same two cone genes, tuned a few tens of"
        " nanometres apart at most. The large steps are between kinds of colour vision, not between species"
        " of one kind."
    ): (
        "Živočich, jehož vidění se simuluje, s typem barevného vidění v závorce."
        "\n\nVětšina dichromatických savců vypadá velmi podobně: mají stejné dva geny pro čípky, naladěné"
        " nanejvýš pár desítek nanometrů od sebe. Velké skoky jsou mezi typy barevného vidění, ne mezi druhy"
        " jednoho typu."
    ),
    (
        "How far the cones adapt to the scene instead of to daylight."
        "\n\nAt 0 the eye is adapted to daylight, so a scene lit by a warm sunset looks warm. At 100 each"
        " cone's signal is scaled so that the scene's average colour becomes neutral, the way an eye that has"
        " been in that light for a while stops noticing its cast (von Kries adaptation to a \"grey world\")."
    ): (
        "Jak moc se čípky přizpůsobí scéně místo dennímu světlu."
        "\n\nPři 0 je oko adaptované na denní světlo, takže scéna osvětlená teplým západem slunce vypadá teple."
        " Při 100 se signál každého čípku přeškáluje tak, aby průměrná barva scény byla neutrální, podobně"
        " jako oko, které je v tom světle už chvíli, přestane jeho nádech vnímat (von Kriesova adaptace na"
        " „šedý svět“)."
    ),
    (
        "Blends the simulation with the original image: 0 shows the original, 100 the full simulation."
        "\n\nIn between, the colours the animal cannot tell apart are only partly merged."
    ): (
        "Míchá simulaci s původním obrazem: 0 ukáže originál, 100 plnou simulaci."
        "\n\nMezi tím barvy, které živočich nerozliší, splynou jen zčásti."
    ),
    (
        "Colours the animal cannot tell apart are merged, and the rest keep the saturation the projection"
        " happens to give them."
        "\n\nThe merging says which colours look alike, but not how vivid the others look: that has no natural"
        " scale. For a trichromat nothing merges, so this leaves the image as it is."
    ): (
        "Barvy, které živočich nerozliší, splynou a ostatní si ponechají sytost, jakou jim projekce zrovna"
        " dá."
        "\n\nSplynutí říká, které barvy vypadají stejně, ale ne, jak živé jsou ostatní: to nemá přirozené"
        " měřítko. U trichromata nesplyne nic, takže obraz zůstane beze změny."
    ),
    (
        "Saturation is rescaled so that a human looking at the image can tell as many colour steps apart as"
        " the animal can in the scene. An animal that tells colours apart worse than we do gets paler"
        " colours."
        "\n\nThe steps are counted by the receptor noise limited model, from the share of each kind of cone;"
        " the factors are under RNL scale in Selected species."
        "\n\nIt assumes the animal's cones are as noisy as ours, and it is matched near grey, so strongly"
        " saturated colours are extrapolated."
    ): (
        "Sytost se přeškáluje tak, aby člověk, který se na obraz dívá, rozlišil stejně barevných kroků, kolik"
        " by jich živočich rozlišil ve scéně. Živočich, který rozlišuje barvy hůř než my, dostane bledší"
        " barvy."
        "\n\nKroky počítá model šumem omezených receptorů (RNL) z podílu jednotlivých druhů čípků; faktory jsou"
        " v řádku Škála RNL v panelu Vybraný druh."
        "\n\nPředpokládá, že čípky živočicha jsou stejně zašuměné jako naše, a je sladěný poblíž šedé, takže"
        " výrazně syté barvy jsou extrapolované."
    ),
    (
        "Removes the detail finer than the species resolves, with a blur matching its measured acuity (Acuity"
        " in Selected species)."
        "\n\nThe blur shows only if the image has more pixels per degree than the species resolves. A camera"
        " image 640 pixels wide spanning 60° has about 11 pixels per degree, so a dog's blur is under half a"
        " pixel and invisible, while a large photo shows it clearly."
    ): (
        "Odstraní detail jemnější, než druh rozliší, rozostřením podle jeho změřené ostrosti (řádek Ostrost v"
        " panelu Vybraný druh)."
        "\n\nRozostření je vidět, jen když má obraz víc pixelů na stupeň, než druh rozliší. Obraz z kamery"
        " široký 640 pixelů, který zabírá 60°, má asi 11 pixelů na stupeň, takže rozostření psa je menší než"
        " půl pixelu a neviditelné, kdežto velká fotka ho ukáže zřetelně."
    ),
    (
        "How many degrees of view the image spans from left to right. The blur is set per degree, so this"
        " decides how many pixels wide it is: halving the angle halves the blur."
        "\n\n60° is typical of a camera. The result is right when the image is seen at that same angle; seen"
        " smaller, your own acuity blurs it further."
    ): (
        "Kolik stupňů zorného pole obraz zabírá zleva doprava. Rozostření se určuje na stupeň, takže tohle"
        " rozhoduje, kolik pixelů je široké: poloviční úhel znamená poloviční rozostření."
        "\n\n60° je obvyklé u kamery. Výsledek sedí, když se na obraz díváte pod stejným úhlem; zmenšený ho"
        " vaše vlastní ostrost rozostří ještě víc."
    ),
    (
        "Shows a second image to the left of the simulation: the original, or another species chosen under"
        " Left image. Unchecked, only the simulation is shown."
    ): (
        "Ukáže vlevo od simulace druhý obraz: originál, nebo jiný druh vybraný v poli Levý obraz."
        " Nezaškrtnuté ukáže jen simulaci."
    ),
    (
        "What the left image shows: the original, or another species rendered with the same settings, so that"
        " two animals can be compared directly."
    ): (
        "Co ukazuje levý obraz: originál, nebo jiný druh vykreslený se stejným nastavením, aby šlo dva"
        " živočichy přímo porovnat."
    ),
    (
        "Adds a third image: grey where the left and right images look the same, red where they differ by"
        " more than one just-noticeable difference, deeper the larger the difference."
        "\n\nIt compares the two images as a human sees them, in CIELAB, where one just-noticeable difference"
        " is about ΔE 2.3 (Mahy et al. 1994), so it catches differences in colour and in sharpness alike."
        "\n\nIt measures the two renderings, each already reduced to what its animal can tell apart. The"
        " threshold is an average, so the edge of the red region is approximate."
    ): (
        "Přidá třetí obraz: šedě tam, kde levý a pravý obraz vypadají stejně, červeně tam, kde se liší o víc"
        " než jeden právě postřehnutelný rozdíl, a tím sytěji, čím je rozdíl větší."
        "\n\nPorovnává oba obrazy tak, jak je vidí člověk, v prostoru CIELAB, kde je jeden právě postřehnutelný"
        " rozdíl asi ΔE 2,3 (Mahy et al. 1994), takže zachytí rozdíly v barvě i v ostrosti."
        "\n\nMěří dvě vykreslení, každé už omezené na to, co jeho živočich rozliší. Práh je průměrný, takže"
        " okraj červené oblasti je přibližný."
    ),
    (
        "Returns every control to the values given on the command line."
    ): (
        "Vrátí všechny ovládací prvky na hodnoty zadané na příkazové řádce."
    ),
    (
        "Saves the images as shown, at the camera's resolution, as dog-<species>-<time>.png in the current"
        " directory."
    ): (
        "Uloží obrazy tak, jak jsou zobrazené, v rozlišení kamery, jako dog-<druh>-<čas>.png do aktuálního"
        " adresáře."
    ),
    (
        "Shows a photo or a video instead of the camera, with every control working on it as on the camera."
        "\n\nA large file is shown scaled down, so that the controls stay quick; Convert file works on it at"
        " full size. A video plays at its own rate and starts over at its end."
    ): (
        "Ukáže místo kamery fotku nebo video; všechny ovládací prvky na ně působí stejně jako na kameru."
        "\n\nVelký soubor se zobrazí zmenšený, aby ovládání zůstalo svižné; Převést soubor s ním pracuje v plné"
        " velikosti. Video se přehrává svou rychlostí a na konci začne znovu."
    ),
    (
        "Shows the camera again instead of the open file."
    ): (
        "Ukáže místo otevřeného souboru znovu kameru."
    ),
    (
        "Converts the open photo or video at full size with the current settings, and writes it next to the"
        " original as <name>.dog.png or <name>.dog.mp4. The result shows what the window shows: side by side,"
        " the other species and the map of differences included."
        "\n\nA video is written as H.265 where the system can, and otherwise with the best codec it has. Its"
        " sound is kept when ffmpeg is installed; without ffmpeg the video comes out silent."
    ): (
        "Převede otevřenou fotku nebo video v plné velikosti se současným nastavením a zapíše je vedle"
        " originálu jako <název>.dog.png nebo <název>.dog.mp4. Výsledek ukazuje totéž co okno: včetně"
        " zobrazení vedle sebe, druhého druhu a mapy rozdílů."
        "\n\nVideo se zapíše jako H.265, kde to systém umí, jinak nejlepším kodekem, který má. Zvuk zůstane"
        " zachovaný, když je nainstalovaný ffmpeg; bez ffmpeg vyjde video bez zvuku."
    ),
}

CZECH_SPECIES = {
    "dog": "pes",
    "cat": "kočka",
    "horse": "kůň",
    "cow": "kráva",
    "sheep": "ovce",
    "goat": "koza",
    "pig": "prase",
    "fallow-deer": "daněk",
    "white-tailed-deer": "jelenec běloocasý",
    "guinea-pig": "morče",
    "tree-squirrel": "veverka",
    "ground-squirrel": "sysel kalifornský",
    "ferret": "fretka",
    "protanope": "protanop",
    "deuteranope": "deuteranop",
    "human": "člověk",
    "protanomalous": "protanomál",
    "deuteranomalous": "deuteranomál",
    "macaque": "makak",
    "howler-monkey": "vřešťan",
    "marmoset-female": "kosman, samice",
    "harbour-seal": "tuleň obecný",
    "bottlenose-dolphin": "delfín skákavý",
}


@dataclasses.dataclass(frozen=True)
class Language:
    name: str  # in the language itself
    decimal_point: str = "."
    texts: dict[str, str] = dataclasses.field(default_factory=dict)  # English text: its translation
    species: dict[str, str] = dataclasses.field(default_factory=dict)  # SPECIES key: its name


# Keyed by ISO 639-1 code, the way locale names start. English needs no catalogue:
# its texts are the keys, and the SPECIES keys its species names.
LANGUAGES = {
    "cs": Language("Čeština", ",", CZECH, CZECH_SPECIES),
    "en": Language("English"),
}


def translate(text: str, language: str) -> str:
    return LANGUAGES[language].texts.get(text, text)


def species_name(species: str, language: str) -> str:
    return LANGUAGES[language].species.get(species, species)


def number(value: float, spec: str, language: str) -> str:
    """format(value, spec), with the language's decimal point."""
    return format(value, spec).replace(".", LANGUAGES[language].decimal_point)


def _language_of(locale_name: str) -> str | None:
    """"en" from "en_GB.UTF-8", if it is one of LANGUAGES."""
    code = locale_name.lower().split("_")[0].split(".")[0].split("@")[0]
    return code if code in LANGUAGES else None


def system_language() -> str:
    """The first of LANGUAGES the user's system asks for, and English if none.

    Like gettext, only the first of LANGUAGE, LC_ALL, LC_MESSAGES and LANG that is set
    counts, and LANGUAGE may list several languages. Windows usually sets none of them,
    so there the user's interface language is asked for instead.
    """
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(variable)
        if value:
            for name in value.split(":"):
                language = _language_of(name)
                if language:
                    return language
            return "en"
    if sys.platform == "win32":
        import ctypes

        name = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    else:
        name = locale.getlocale()[0] or ""
    return _language_of(name) or "en"
