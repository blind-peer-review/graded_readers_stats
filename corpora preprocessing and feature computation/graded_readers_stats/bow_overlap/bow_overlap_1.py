###############################################################################
# Goal: Check the overlap between BOW output and vocabulary terms.
###############################################################################
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from graded_readers_stats.data import read_pandas_csv


class Level(Enum):
    LOW = 1
    MID = 2
    HIGH = 3
    MIXED = 4


@dataclass
class Bundle:
    name: str
    items: set[str]
    level: Level


def print_overlap(
        header: str,
        words_vocab: set[str],
        words_bow: set[str]
) -> None:
    print("---")
    print(header)
    words_overlap = words_vocab.intersection(words_bow)
    print("Overlap count: ", len(words_overlap))
    print(sorted(words_overlap))


###############################################################################
# DATA
###############################################################################

vocabulary_csv_path = "../../data/vocabulary.csv"
vocab_df = read_pandas_csv(vocabulary_csv_path)
vocab_df["Lexical item"] = vocab_df["Lexical item"].apply(
    lambda x: x.lower().strip()
)

vocab_all_words = set(vocab_df["Lexical item"])
vocab_level1_words = set(vocab_df[vocab_df["Level"] == "A1-A2"]["Lexical item"])
vocab_level2_words = set(vocab_df[vocab_df["Level"] == "B1"]["Lexical item"])
vocab_level3_words = set(vocab_df[vocab_df["Level"] == "B2"]["Lexical item"])

# Graded
bow_graded_1_all = {
    "mucho": 0.270270,
    "ser": 0.266858,
    "no": 0.237320,
    "estar": 0.192712,
    "camping": 0.170497,
    "decir": 0.167421,
    "uno": 0.163530,
    "cuadro": 0.131144,
    "contesto": 0.129713,
    "ir": 0.124620,
    "lunes": 0.123941,
    "agencia": 0.122170,
    "tener": 0.121078,
    "sinagoga": 0.119608,
    "verónicamente": 0.114235,
    "español": 0.112168,
    "gustar": 0.103373,
    "vázquez": 0.102195,
    "ladrón": 0.096983,
    "inspector": 0.095439,
    "billete": 0.095237,
    "señor": 0.090846,
    "museo": 0.085734,
    "qué": 0.083902,
    "sí": 0.083464,
    "saber": 0.083171,
    "pregunta": 0.082940,
    "día": 0.082688,
    "leer": 0.081594,
    "llave": 0.079560,
    "ahora": 0.079168,
    "aquí": 0.079120,
    "vacaciones": 0.078584,
    "ventanilla": 0.077859,
    "trabajar": 0.075808,
    "socio": 0.074175,
    "caso": 0.074131,
    "volar": 0.073857,
    "mañana": 0.073773,
    "interesante": 0.071675,
    "pensar": 0.071661,
    "emplear": 0.068735,
    "fuente": 0.068724,
    "querer": 0.068565,
    "artículo": 0.067524,
    "escribir": 0.065634,
    "quién": 0.065199,
    "chocolate": 0.063599,
    "azul": 0.063594,
    "buen": 0.063034,
    "poder": 0.059872,
    "inscripción": 0.059804,
    "amigo": 0.059709,
    "semana": 0.059113,
    "café": 0.059077,
    "hoy": 0.057876,
    "entender": 0.057304,
    "cano": 0.056832,
    "jacinto": 0.056832,
    "hucho": 0.056506,
    "que": -0.522539,
    "el": -0.147248,
    "con": -0.117295,
    "de": -0.107501,
    "como": -0.086297,
    "él": -0.084743,
    "pueblo": -0.077594,
    "balsa": -0.076481,
    "mujer": -0.069517,
    "marinero": -0.068905,
    "tú": -0.066161,
    "mar": -0.065775,
    "disco": -0.065582,
    "si": -0.064093,
    "su": -0.061466,
    "yo": -0.058638,
    "fuego": -0.053273,
    "propio": -0.050257,
    "pez": -0.049644,
    "ropa": -0.049576,
    "para": -0.049382,
    "aunque": -0.044313,
    "ver": -0.043682,
    "departamento": -0.043069,
    "rambla": -0.041166,
    "contra": -0.041034,
    "batalla": -0.040687,
    "music": -0.040646,
    "hall": -0.040646,
    "gratin": -0.040646,
    "aquel": -0.040532,
    "fuerza": -0.039497,
    "pronto": -0.038839,
    "religioso": -0.038742,
    "jesuita": -0.038742,
    "empresa": -0.037981,
    "responder": -0.037560,
    "donde": -0.036886,
    "multitud": -0.036808,
    "moto": -0.036735,
    "forma": -0.036061,
    "hombre": -0.035987,
    "embargo": -0.035671,
    "tíaú": -0.035372,
    "zapato": -0.035367,
    "camión": -0.034875,
    "porque": -0.034488,
    "isla": -0.033330,
    "cuando": -0.033027,
    "pedro": -0.032913,
    "marido": -0.032589,
    "brasa": -0.032498,
    "suyo": -0.032075,
    "lanzar": -0.032040,
    "fondo": -0.031636,
    "varios": -0.031450,
    "leo": -0.031323,
    "hans": -0.031323,
    "padre": -0.031252,
    "pieza": -0.031243,
}
bow_graded_1_pos = {k: v for k, v in bow_graded_1_all.items() if v > 0}
bow_graded_1_neg = {k: v for k, v in bow_graded_1_all.items() if v < 0}
bow_graded_2_all = {
    "yo": 0.312155,
    "el": 0.252430,
    "mi": 0.169581,
    "marinero": 0.136704,
    "mujer": 0.116255,
    "de": 0.111748,
    "fuego": 0.099720,
    "pez": 0.099690,
    "batalla": 0.081059,
    "marido": 0.080579,
    "rambla": 0.077952,
    "que": 0.077595,
    "vecino": 0.075380,
    "ropa": 0.075161,
    "multitud": 0.073599,
    "con": 0.071286,
    "río": 0.071202,
    "camión": 0.069479,
    "pedro": 0.066545,
    "brasa": 0.066247,
    "caña": 0.061843,
    "alcalde": 0.057737,
    "san": 0.056722,
    "contra": 0.055909,
    "quemar": 0.054927,
    "hoguera": 0.052998,
    "pueblo": 0.051296,
    "cámara": 0.050675,
    "tienda": 0.050319,
    "hora": 0.050097,
    "abierto": 0.049888,
    "brazo": 0.048297,
    "padre": 0.048143,
    "horchata": 0.046771,
    "añadir": 0.046731,
    "lleno": 0.046174,
    "proteger": 0.045664,
    "empezar": 0.045553,
    "mar": 0.045534,
    "pescar": 0.045314,
    "plaza": 0.044872,
    "lanzar": 0.044630,
    "balcón": 0.044628,
    "nuestro": 0.044592,
    "rey": 0.043710,
    "hacia": 0.043345,
    "susana": 0.042679,
    "hacer": 0.042531,
    "camiseta": 0.042027,
    "allá": 0.041870,
    "alrededor": 0.040185,
    "piscina": 0.039997,
    "cura": 0.039748,
    "manrique": 0.039748,
    "respondí": 0.039744,
    "joven": 0.039456,
    "secuestrado": 0.039447,
    "timbre": 0.039413,
    "dentro": 0.039285,
    "celebrar": 0.039141,
    "no": -0.255535,
    "ser": -0.205412,
    "mucho": -0.156500,
    "tú": -0.147922,
    "decir": -0.103757,
    "sí": -0.102988,
    "gustar": -0.092850,
    "señor": -0.092426,
    "saber": -0.089344,
    "camping": -0.087583,
    "balsa": -0.085070,
    "ese": -0.084439,
    "contesto": -0.078363,
    "haber": -0.077087,
    "pregunta": -0.072332,
    "querer": -0.070971,
    "cuadro": -0.068637,
    "uno": -0.063759,
    "ladrón": -0.062625,
    "sinagoga": -0.061228,
    "agencia": -0.059646,
    "lunes": -0.058975,
    "español": -0.058790,
    "interesar": -0.058347,
    "verónicamente": -0.057134,
    "cosa": -0.055864,
    "inspector": -0.055784,
    "tener": -0.054076,
    "ahora": -0.052864,
    "buen": -0.052799,
    "trabajar": -0.052269,
    "bueno": -0.052208,
    "aquí": -0.051411,
    "arte": -0.049806,
    "vacaciones": -0.048952,
    "entender": -0.048910,
    "ciudad": -0.048576,
    "poner": -0.047980,
    "extranjero": -0.047198,
    "hoy": -0.046407,
    "hablar": -0.045887,
    "pequeño": -0.044378,
    "qué": -0.044265,
    "billete": -0.044229,
    "music": -0.042935,
    "hall": -0.042935,
    "gratin": -0.042935,
    "vázquez": -0.042835,
    "semana": -0.042249,
    "interesante": -0.042165,
    "religioso": -0.041557,
    "jesuita": -0.041557,
    "artículo": -0.041388,
    "robo": -0.041296,
    "comprar": -0.041143,
    "museo": -0.039730,
    "claro": -0.039168,
    "ventanilla": -0.039157,
    "su": -0.038736,
    "ir": -0.038114,
}
bow_graded_2_pos = {k: v for k, v in bow_graded_2_all.items() if v > 0}
bow_graded_2_neg = {k: v for k, v in bow_graded_2_all.items() if v < 0}
bow_graded_3_all = {
    "que": 0.444945,
    "tú": 0.214084,
    "balsa": 0.161551,
    "disco": 0.103246,
    "su": 0.100201,
    "gratin": 0.083581,
    "hall": 0.083581,
    "music": 0.083581,
    "departamento": 0.081026,
    "jesuita": 0.080299,
    "religioso": 0.080299,
    "él": 0.078292,
    "propio": 0.075726,
    "como": 0.071720,
    "tíaú": 0.066384,
    "forma": 0.063618,
    "cosa": 0.062349,
    "indio": 0.062346,
    "guaraníe": 0.060224,
    "industria": 0.060224,
    "para": 0.059985,
    "hans": 0.058928,
    "leo": 0.058928,
    "velar": 0.058746,
    "pronto": 0.058166,
    "porque": 0.057684,
    "nativo": 0.057056,
    "social": 0.054855,
    "querío": 0.054276,
    "si": 0.053936,
    "arte": 0.053635,
    "europeo": 0.053532,
    "grupo": 0.050665,
    "belle": 0.050149,
    "dama": 0.050149,
    "mantener": 0.050149,
    "époque": 0.050149,
    "tu": 0.049984,
    "indígena": 0.048784,
    "americano": 0.048745,
    "sistema": 0.048178,
    "artístico": 0.047602,
    "aunque": 0.046687,
    "hombre": 0.046619,
    "con": 0.046008,
    "italiano": 0.045736,
    "niña": 0.045641,
    "cuyo": 0.045182,
    "expedición": 0.044965,
    "embarcación": 0.044059,
    "polinesio": 0.044059,
    "tronco": 0.044059,
    "civilización": 0.043991,
    "obra": 0.043967,
    "nombre": 0.043642,
    "colegio": 0.043101,
    "cuando": 0.042650,
    "personal": 0.041814,
    "moto": 0.041798,
    "antepasado": 0.041003,
    "yo": -0.253517,
    "estar": -0.207048,
    "mi": -0.159662,
    "mucho": -0.113770,
    "el": -0.105182,
    "uno": -0.099771,
    "ir": -0.086506,
    "camping": -0.082915,
    "marinero": -0.067799,
    "tener": -0.067002,
    "lunes": -0.064966,
    "decir": -0.063664,
    "agencia": -0.062525,
    "cuadro": -0.062507,
    "ser": -0.061446,
    "vázquez": -0.059360,
    "sinagoga": -0.058379,
    "hacer": -0.057402,
    "verónicamente": -0.057101,
    "bar": -0.056432,
    "dentro": -0.055261,
    "español": -0.053378,
    "ventana": -0.053142,
    "chico": -0.052481,
    "sol": -0.052166,
    "cerca": -0.052062,
    "plaza": -0.051381,
    "contesto": -0.051350,
    "calor": -0.051211,
    "billete": -0.051008,
    "coger": -0.050234,
    "aire": -0.050086,
    "pez": -0.050047,
    "día": -0.049793,
    "joya": -0.049745,
    "pensar": -0.048801,
    "fiesta": -0.048433,
    "alcalde": -0.048324,
    "marido": -0.047990,
    "mujer": -0.046737,
    "caso": -0.046666,
    "mañana": -0.046641,
    "vecino": -0.046557,
    "fuego": -0.046447,
    "río": -0.046333,
    "rey": -0.046324,
    "museo": -0.046004,
    "nuevo": -0.045047,
    "ojo": -0.044834,
    "leer": -0.044419,
    "dos": -0.043917,
    "entrar": -0.043839,
    "llave": -0.043643,
    "allí": -0.041813,
    "sentar": -0.041680,
    "volar": -0.041382,
    "nadie": -0.041330,
    "batalla": -0.040372,
    "después": -0.040245,
    "inspector": -0.039656,
}
bow_graded_3_pos = {k: v for k, v in bow_graded_3_all.items() if v > 0}
bow_graded_3_neg = {k: v for k, v in bow_graded_3_all.items() if v < 0}
bow_graded_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all)
)
bow_graded_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos)
)
bow_graded_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg)
)

# Literature
bow_litera_1_all = {
    "leotoldo": 0.186946,
    "día": 0.139741,
    "fábrica": 0.120534,
    "poder": 0.111920,
    "niña": 0.108387,
    "bosque": 0.103753,
    "para": 0.103107,
    "príncipe": 0.097190,
    "bruja": 0.093306,
    "casa": 0.092700,
    "que": 0.092391,
    "aprender": 0.083771,
    "cada": 0.079967,
    "lugar": 0.079870,
    "gato": 0.079853,
    "lámpara": 0.079496,
    "invento": 0.078664,
    "volar": 0.077418,
    "nuevo": 0.077403,
    "nuestro": 0.073321,
    "tres": 0.073248,
    "así": 0.071879,
    "ricito": 0.069795,
    "vivir": 0.069034,
    "barco": 0.066137,
    "patito": 0.065313,
    "encontrar": 0.063548,
    "cuento": 0.063447,
    "mucho": 0.061784,
    "contar": 0.060833,
    "osa": 0.060740,
    "polar": 0.060740,
    "planeta": 0.059567,
    "pan": 0.058930,
    "entrenar": 0.058791,
    "él": 0.058548,
    "mayor": 0.058446,
    "amarillo": 0.056395,
    "convertir": 0.055915,
    "carabás": 0.054724,
    "marqués": 0.054724,
    "cole": 0.054080,
    "coronavir": 0.054080,
    "allí": 0.054050,
    "compartir": 0.052994,
    "balleno": 0.052788,
    "pato": 0.050527,
    "comenzar": 0.050335,
    "descubrir": 0.049668,
    "hada": 0.049384,
    "al": 0.049190,
    "salir": 0.048900,
    "animal": 0.048701,
    "amar": 0.046335,
    "junto": 0.046135,
    "cosa": 0.045031,
    "perdido": 0.044709,
    "donde": 0.044606,
    "virus": 0.044094,
    "chocolate": 0.043692,
    "yo": -0.496210,
    "el": -0.269072,
    "de": -0.257279,
    "no": -0.254222,
    "mi": -0.226799,
    "decir": -0.226759,
    "haber": -0.160109,
    "ser": -0.159675,
    "rosa": -0.150013,
    "en": -0.140230,
    "cohete": -0.138970,
    "molinero": -0.134810,
    "gigante": -0.131886,
    "muñeco": -0.129824,
    "pero": -0.109842,
    "rojo": -0.109718,
    "ruiseñor": -0.109419,
    "jardín": -0.105910,
    "qué": -0.086504,
    "pues": -0.083296,
    "preguntar": -0.078389,
    "carretilla": -0.068174,
    "responder": -0.066513,
    "sobre": -0.065745,
    "rosal": -0.063886,
    "estudiante": -0.063328,
    "gritar": -0.061857,
    "tener": -0.060876,
    "árbol": -0.060852,
    "pelo": -0.060794,
    "ese": -0.060132,
    "replicar": -0.057940,
    "señor": -0.056455,
    "exclamar": -0.053124,
    "mano": -0.052902,
    "mismo": -0.052863,
    "habitación": -0.050412,
    "creer": -0.049632,
    "blanco": -0.049348,
    "hacia": -0.048708,
    "embargo": -0.046489,
    "cabeza": -0.046027,
    "brazo": -0.044111,
    "ojo": -0.043238,
    "pardillo": -0.043057,
    "sristino": -0.038473,
    "verdaderamente": -0.038472,
    "espino": -0.038083,
    "rata": -0.037825,
    "realmente": -0.037746,
    "gran": -0.037428,
    "capullo": -0.037059,
    "dos": -0.036854,
    "rubí": -0.036482,
    "estatua": -0.036077,
    "corazón": -0.035812,
    "hablar": -0.035073,
    "contra": -0.034787,
    "candela": -0.034743,
    "egoísta": -0.034503,
}
bow_litera_1_pos = {k: v for k, v in bow_litera_1_all.items() if v > 0}
bow_litera_1_neg = {k: v for k, v in bow_litera_1_all.items() if v < 0}
bow_litera_2_all = {
    "el": 0.816768,
    "yo": 0.260393,
    "decir": 0.177565,
    "rosa": 0.170498,
    "mi": 0.164313,
    "molinero": 0.159359,
    "cohete": 0.157521,
    "gigante": 0.150813,
    "muñeco": 0.147875,
    "rojo": 0.123375,
    "jardín": 0.122563,
    "ruiseñor": 0.122170,
    "no": 0.110276,
    "carretilla": 0.077178,
    "señor": 0.074287,
    "rosal": 0.071428,
    "gritar": 0.070968,
    "pelo": 0.068766,
    "pero": 0.066787,
    "sobre": 0.066067,
    "pues": 0.065054,
    "estudiante": 0.064498,
    "árbol": 0.063934,
    "ser": 0.061069,
    "preguntar": 0.058735,
    "replicar": 0.057567,
    "blanco": 0.055508,
    "mano": 0.055474,
    "cabeza": 0.055128,
    "exclamar": 0.053993,
    "niño": 0.053554,
    "pardillo": 0.048744,
    "espino": 0.046374,
    "habitación": 0.046232,
    "gran": 0.045017,
    "hacia": 0.044851,
    "responder": 0.043909,
    "sristino": 0.043827,
    "realmente": 0.042084,
    "rata": 0.041308,
    "capullo": 0.041082,
    "rubí": 0.040678,
    "estatua": 0.039399,
    "candela": 0.039380,
    "coger": 0.038559,
    "primavera": 0.038358,
    "corazón": 0.038194,
    "egoísta": 0.037687,
    "ojo": 0.037281,
    "padre": 0.037037,
    "hombro": 0.036586,
    "amistad": 0.036415,
    "brazo": 0.036405,
    "qué": 0.036156,
    "verdaderamente": 0.035855,
    "tener": 0.034540,
    "creer": 0.034510,
    "flor": 0.034076,
    "entonces": 0.033213,
    "cantar": 0.033128,
    "que": -0.278499,
    "él": -0.182535,
    "leotoldo": -0.139707,
    "día": -0.117536,
    "poder": -0.112349,
    "uno": -0.097393,
    "para": -0.093925,
    "casa": -0.088157,
    "fábrica": -0.085488,
    "con": -0.083917,
    "niña": -0.080596,
    "bosque": -0.078744,
    "bruja": -0.075253,
    "tú": -0.074256,
    "príncipe": -0.071735,
    "lugar": -0.068602,
    "nuestro": -0.066628,
    "cada": -0.066434,
    "aprender": -0.065820,
    "este": -0.064013,
    "nuevo": -0.063945,
    "gato": -0.063236,
    "encontrar": -0.062547,
    "lámpara": -0.060732,
    "así": -0.060172,
    "invento": -0.058742,
    "por": -0.057277,
    "volar": -0.057180,
    "vivir": -0.055571,
    "barco": -0.054673,
    "hacer": -0.054286,
    "tres": -0.054184,
    "ricito": -0.053763,
    "mayor": -0.052129,
    "mucho": -0.048498,
    "cuento": -0.047741,
    "patito": -0.047589,
    "pan": -0.047208,
    "comenzar": -0.047058,
    "planeta": -0.046665,
    "contar": -0.046377,
    "polar": -0.046239,
    "osa": -0.046239,
    "entrenar": -0.045249,
    "también": -0.043569,
    "coronavir": -0.043283,
    "cole": -0.043283,
    "marqués": -0.043093,
    "carabás": -0.043093,
    "convertir": -0.042980,
    "salir": -0.042699,
    "allí": -0.042652,
    "hombre": -0.042608,
    "hada": -0.040650,
    "mujer": -0.040472,
    "balleno": -0.039653,
    "cosa": -0.039153,
    "descubrir": -0.039122,
    "al": -0.038840,
    "animal": -0.038779,
}
bow_litera_2_pos = {k: v for k, v in bow_litera_2_all.items() if v > 0}
bow_litera_2_neg = {k: v for k, v in bow_litera_2_all.items() if v < 0}
bow_litera_3_all = {
    "de": 0.294330,
    "yo": 0.235816,
    "que": 0.186107,
    "haber": 0.152332,
    "no": 0.143946,
    "en": 0.143278,
    "él": 0.123987,
    "tú": 0.105900,
    "ser": 0.098605,
    "ese": 0.086912,
    "por": 0.068680,
    "este": 0.068086,
    "uno": 0.063981,
    "mi": 0.062486,
    "qué": 0.050349,
    "decir": 0.049194,
    "con": 0.046074,
    "carta": 0.044680,
    "pero": 0.043054,
    "mismo": 0.037521,
    "cierto": 0.036594,
    "palabra": 0.035055,
    "sin": 0.032007,
    "embargo": 0.029204,
    "modo": 0.028858,
    "si": 0.027756,
    "después": 0.027678,
    "pensar": 0.027540,
    "tomar": 0.027472,
    "como": 0.027160,
    "instante": 0.026986,
    "fué": 0.026804,
    "tener": 0.026336,
    "recuerdo": 0.026073,
    "vez": 0.026003,
    "aquel": 0.025718,
    "otro": 0.025695,
    "saber": 0.025529,
    "parecer": 0.025421,
    "estancia": 0.025142,
    "tiempo": 0.024958,
    "momento": 0.024705,
    "alguno": 0.023570,
    "pedro": 0.023368,
    "hablar": 0.023096,
    "tal": 0.022922,
    "hacer": 0.022863,
    "idea": 0.022792,
    "responder": 0.022604,
    "pregunta": 0.022543,
    "escena": 0.020784,
    "rublo": 0.020447,
    "algo": 0.020079,
    "hombre": 0.019771,
    "preguntar": 0.019654,
    "deber": 0.019550,
    "advertir": 0.019090,
    "sí": 0.018877,
    "posibilidad": 0.018426,
    "mirada": 0.018383,
    "el": -0.547696,
    "leotoldo": -0.047239,
    "niño": -0.038060,
    "fábrica": -0.035046,
    "niña": -0.027791,
    "príncipe": -0.025455,
    "bosque": -0.025009,
    "pequeño": -0.024891,
    "molinero": -0.024550,
    "amarillo": -0.023639,
    "día": -0.022205,
    "compartir": -0.020934,
    "rosa": -0.020485,
    "volar": -0.020239,
    "invento": -0.019921,
    "tres": -0.019063,
    "gigante": -0.018927,
    "lámpara": -0.018764,
    "cohete": -0.018550,
    "bruja": -0.018053,
    "muñeco": -0.018051,
    "aprender": -0.017951,
    "señor": -0.017832,
    "patito": -0.017724,
    "padre": -0.017412,
    "jardín": -0.016653,
    "gato": -0.016618,
    "rey": -0.016256,
    "ricito": -0.016032,
    "cuento": -0.015706,
    "polar": -0.014501,
    "osa": -0.014501,
    "contar": -0.014456,
    "feliz": -0.014385,
    "agua": -0.014126,
    "gustar": -0.013892,
    "voló": -0.013845,
    "rojo": -0.013656,
    "pato": -0.013547,
    "junto": -0.013543,
    "entrenar": -0.013542,
    "cada": -0.013533,
    "vivir": -0.013462,
    "nuevo": -0.013458,
    "jugar": -0.013382,
    "mucho": -0.013286,
    "balleno": -0.013135,
    "convertir": -0.012935,
    "oro": -0.012918,
    "planeta": -0.012902,
    "ruiseñor": -0.012750,
    "madera": -0.012667,
    "princesa": -0.012195,
    "joven": -0.011898,
    "pan": -0.011722,
    "así": -0.011707,
    "marqués": -0.011632,
    "carabás": -0.011632,
    "bello": -0.011538,
    "barco": -0.011464,
}
bow_litera_3_pos = {k: v for k, v in bow_litera_3_all.items() if v > 0}
bow_litera_3_neg = {k: v for k, v in bow_litera_3_all.items() if v < 0}
bow_litera_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_litera_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_litera_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)

# Graded + Literature
bow_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all,
     bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos,
     bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg,
     bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)

###############################################################################
# BUNDLES
###############################################################################

# Vocabulary
bundle_vocab_level_1 = Bundle(
    name="Vocab Level 1",
    items=vocab_level1_words,
    level=Level.LOW
)
bundle_vocab_level_2 = Bundle(
    name="Vocab Level 2",
    items=vocab_level2_words,
    level=Level.MID
)
bundle_vocab_level_3 = Bundle(
    name="Vocab Level 3",
    items=vocab_level3_words,
    level=Level.HIGH
)
bundle_vocab_all = Bundle(
    name="Vocab All Levels",
    items=vocab_all_words,
    level=Level.MIXED
)

# Graded Level 1
bundle_graded_1_all = Bundle(
    name="All Graded Level 1",
    items=set(bow_graded_1_all.keys()),
    level=Level.LOW
)
bundle_graded_1_pos = Bundle(
    name="Positive Graded Level 1",
    items=set(bow_graded_1_pos.keys()),
    level=Level.LOW
)
bundle_graded_1_neg = Bundle(
    name="Negative Graded Level 1",
    items=set(bow_graded_1_neg.keys()),
    level=Level.LOW
)

# Graded Level 2
bundle_graded_2_all = Bundle(
    name="All Graded Level 2",
    items=set(bow_graded_2_all.keys()),
    level=Level.LOW
)
bundle_graded_2_pos = Bundle(
    name="Positive Graded Level 2",
    items=set(bow_graded_2_pos.keys()),
    level=Level.LOW
)
bundle_graded_2_neg = Bundle(
    name="Negative Graded Level 2",
    items=set(bow_graded_2_neg.keys()),
    level=Level.LOW
)

# Graded Level 3
bundle_graded_3_all = Bundle(
    name="All Graded Level 3",
    items=set(bow_graded_3_all.keys()),
    level=Level.LOW
)
bundle_graded_3_pos = Bundle(
    name="Positive Graded Level 3",
    items=set(bow_graded_3_pos.keys()),
    level=Level.LOW
)
bundle_graded_3_neg = Bundle(
    name="Negative Graded Level 3",
    items=set(bow_graded_3_neg.keys()),
    level=Level.LOW
)

# Graded All
bundle_graded_all = Bundle(
    name="All Graded",
    items=set(bow_graded_all.keys()),
    level=Level.LOW
)
bundle_graded_pos = Bundle(
    name="Positive Graded",
    items=set(bow_graded_pos.keys()),
    level=Level.LOW
)
bundle_graded_neg = Bundle(
    name="Negative Graded",
    items=set(bow_graded_neg.keys()),
    level=Level.LOW
)

# Litera Level 1
bundle_litera_1_all = Bundle(
    name="All Litera Level 1",
    items=set(bow_litera_1_all.keys()),
    level=Level.LOW
)
bundle_litera_1_pos = Bundle(
    name="Positive Litera Level 1",
    items=set(bow_litera_1_pos.keys()),
    level=Level.LOW
)
bundle_litera_1_neg = Bundle(
    name="Negative Litera Level 1",
    items=set(bow_litera_1_neg.keys()),
    level=Level.LOW
)

# Litera Level 2
bundle_litera_2_all = Bundle(
    name="All Litera Level 2",
    items=set(bow_litera_2_all.keys()),
    level=Level.LOW
)
bundle_litera_2_pos = Bundle(
    name="Positive Litera Level 2",
    items=set(bow_litera_2_pos.keys()),
    level=Level.LOW
)
bundle_litera_2_neg = Bundle(
    name="Negative Litera Level 2",
    items=set(bow_litera_2_neg.keys()),
    level=Level.LOW
)

# Litera Level 3
bundle_litera_3_all = Bundle(
    name="All Litera Level 3",
    items=set(bow_litera_3_all.keys()),
    level=Level.LOW
)
bundle_litera_3_pos = Bundle(
    name="Positive Litera Level 3",
    items=set(bow_litera_3_pos.keys()),
    level=Level.LOW
)
bundle_litera_3_neg = Bundle(
    name="Negative Litera Level 3",
    items=set(bow_litera_3_neg.keys()),
    level=Level.LOW
)

# Litera All
bundle_litera_all = Bundle(
    name="All Litera",
    items=set(bow_litera_all.keys()),
    level=Level.LOW
)
bundle_litera_pos = Bundle(
    name="Positive Litera",
    items=set(bow_litera_pos.keys()),
    level=Level.LOW
)
bundle_litera_neg = Bundle(
    name="Negative Litera",
    items=set(bow_litera_neg.keys()),
    level=Level.LOW
)

# Lists of Bundles
all_vocab_bundles = [
    bundle_vocab_level_1,
    bundle_vocab_level_2,
    bundle_vocab_level_3,
    bundle_vocab_all,
]
all_text_bundles = [
    bundle_graded_1_all,
    bundle_graded_1_pos,
    bundle_graded_1_neg,
    bundle_graded_2_all,
    bundle_graded_2_pos,
    bundle_graded_2_neg,
    bundle_graded_3_all,
    bundle_graded_3_pos,
    bundle_graded_3_neg,
    bundle_graded_all,
    bundle_graded_pos,
    bundle_graded_neg,
    bundle_litera_1_all,
    bundle_litera_1_pos,
    bundle_litera_1_neg,
    bundle_litera_2_all,
    bundle_litera_2_pos,
    bundle_litera_2_neg,
    bundle_litera_3_all,
    bundle_litera_3_pos,
    bundle_litera_3_neg,
    bundle_litera_all,
    bundle_litera_pos,
    bundle_litera_neg,
]

##############################################################################
# Print overlaps (all combinations)
##############################################################################

for bundle_vocab in all_vocab_bundles:
    for bundle_text in all_text_bundles:
        print_overlap(
            header=f"[{bundle_vocab.name}] x [{bundle_text.name}]",
            words_vocab=bundle_vocab.items,
            words_bow=bundle_text.items
        )

##############################################################################
# WordCloud generation
##############################################################################

from os import path
from PIL import Image
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

for i in range(1, 4):
    for prefix in ("ti", "to", "bi", "bo"):
        mask = np.array(Image.open(path.join(d, f"masks/{prefix}{i}.jpg")))
        stopwords = set(STOPWORDS)
        stopwords.add("said")
        wc = WordCloud(
            mode="RGBA",
            background_color=None,
            width=700,
            height=700,
            max_words=50,
            mask=mask,
            stopwords=stopwords
        )
        wc.generate_from_frequencies(bow_graded_1_all)
        wc.to_file(path.join(d, f"output/{prefix}{i}.png"))


print("Done!")
