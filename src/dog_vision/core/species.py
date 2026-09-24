"""What is known of each species: its cone peaks, the share of its S cones and its acuity, with sources."""

# Photopigment peak wavelengths in nm.
HUMAN_CONES = {"S": 420.7, "M": 530.3, "L": 558.9}  # Stockman & Sharpe (2000)

# Cone peaks in nm: (S, M, L) for a trichromat, (S, L) for a dichromat, (L,) for a
# cone monochromat. Species with an ultraviolet cone (mice, rats, birds) are left
# out: an RGB camera records nothing of what that cone sees.
SPECIES = {
    "dog": (429.0, 555.0),
    "cat": (450.0, 550.0),
    "horse": (428.0, 539.0),
    "cow": (451.3, 555.3),
    "sheep": (445.3, 552.2),
    "goat": (443.3, 552.5),
    "pig": (440.7, 556.7),
    "fallow-deer": (453.6, 542.2),
    "white-tailed-deer": (456.0, 536.8),
    "guinea-pig": (429.0, 529.0),
    "tree-squirrel": (444.0, 543.0),
    "ground-squirrel": (436.7, 518.9),  # the California species
    "ferret": (430.0, 558.0),
    "protanope": (HUMAN_CONES["S"], HUMAN_CONES["M"]),  # human lacking L cones
    "deuteranope": (HUMAN_CONES["S"], HUMAN_CONES["L"]),  # human lacking M cones
    "human": tuple(HUMAN_CONES.values()),
    # Anomalous trichromats of moderate severity: one cone shifted 10 nm towards the
    # other, where Machado, Oliveira & Fernandes (2009) take 20 nm as dichromacy.
    "protanomalous": (HUMAN_CONES["S"], HUMAN_CONES["M"], HUMAN_CONES["L"] - 10),
    "deuteranomalous": (HUMAN_CONES["S"], HUMAN_CONES["M"] + 10, HUMAN_CONES["L"]),
    "macaque": (431.0, 536.0, 565.0),
    "howler-monkey": (430.0, 530.0, 562.0),
    "marmoset-female": (423.0, 543.0, 563.0),  # with the 543 and 563 nm M/L alleles
    "harbour-seal": (510.0,),
    "bottlenose-dolphin": (524.0,),  # the L opsin
}

# Where each species' cone peaks come from.
PEAKS_FROM = {
    "dog": "Neitz, Geist & Jacobs 1989",
    "cat": "Guenther & Zrenner 1993",
    "horse": "Carroll et al. 2001",
    **dict.fromkeys(["cow", "sheep", "goat", "pig", "fallow-deer", "white-tailed-deer"], "Jacobs, Deegan & Neitz 1998"),
    "guinea-pig": "Jacobs & Deegan 1994",
    "tree-squirrel": "Blakeslee, Jacobs & Neitz 1988",
    "ground-squirrel": "Jacobs, Neitz & Crognale 1985",
    "ferret": "Calderone & Jacobs 2003",
    **dict.fromkeys(["protanope", "deuteranope", "human"], "Stockman & Sharpe 2000"),
    "protanomalous": "L shifted 10 nm, see text",
    "deuteranomalous": "M shifted 10 nm, see text",
    "macaque": "Bowmaker et al. 1978, 1991",
    "howler-monkey": "Jacobs et al. 1996, S assumed",
    "marmoset-female": "Travis 1988, Williams 1992",
    "harbour-seal": "Crognale et al. 1998",
    "bottlenose-dolphin": "Fasick et al. 1998",
}

# Share of S cones among all cones, as the (lowest, highest) reported across the
# retina, and where it comes from; the RNL scale uses the middle of the range.
# A species missing here gets ASSUMED_S_CONE_FRACTION.
S_CONE_FRACTION = {
    "dog": (0.10, 0.18, "Mowat et al. 2008"),  # area centralis, periphery
    "cat": (0.10, 0.20, "Linberg et al. 2001"),
    "horse": (0.10, 0.25, "Sandmann et al. 1996"),
    **dict.fromkeys(["cow", "sheep", "pig"], (0.05, 0.10, "Schiviz et al. 2008")),
    "ground-squirrel": (1 / 15, 1 / 15, "Kryger et al. 1998"),  # 14 M cones per S cone
    "ferret": (1 / 15, 1 / 15, "Calderone & Jacobs 2003"),  # 14 L cones per S cone
    **dict.fromkeys(
        ["protanope", "deuteranope", "human", "protanomalous", "deuteranomalous"], (0.08, 0.12, "Curcio et al. 1991")
    ),
}
ASSUMED_S_CONE_FRACTION = (0.10, 0.10)  # the middle of what the measured species span

# L cones per M cone for a trichromat. Humans with normal colour vision range from
# about twice as many L as M cones to the reverse (Roorda & Williams 1999), so the
# RNL scale takes 1:1 for every trichromat and calls it assumed.
ASSUMED_L_TO_M = 1.0

# Visual acuity in cycles per degree, as (resolving detail side by side, resolving
# detail one above another) — the two differ only for cattle, whose pupil is a
# horizontal oval — and where it comes from. A species missing here was not found
# measured, and is left sharp.
HUMAN_ACUITY = ((72.0, 72.0), "Land & Nilsson 2012")
ACUITY = {
    "dog": ((11.6, 11.6), "Odom et al. 1983"),
    "cat": ((10.0, 10.0), "Wässle 1971"),
    "horse": ((23.3, 23.3), "Timney & Keil 1992"),
    "cow": ((2.6, 1.6), "Rehkämper et al. 2000"),
    "sheep": ((12.85, 12.85), "Sumita et al. 2013, 11.7 to 14"),
    "tree-squirrel": ((2.8, 2.8), "Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8"),
    "ground-squirrel": ((4.0, 4.0), "Jacobs et al. 1980"),
    **dict.fromkeys(["protanope", "deuteranope", "human", "protanomalous", "deuteranomalous"], HUMAN_ACUITY),
    "macaque": ((60.0, 60.0), "Nature Neuroscience 2024, about 60"),
    "marmoset-female": ((30.0, 30.0), "Troilo, Howland & Judge 1993"),
    "harbour-seal": ((5.5, 5.5), "Hanke & Dehnhardt 2009, in air"),
    "bottlenose-dolphin": ((60 / (2 * 8.2), 60 / (2 * 8.2)), "Herman et al. 1975, 8.2 arcmin stripes"),
}
