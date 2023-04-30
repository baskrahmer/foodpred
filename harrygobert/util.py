import argparse
import os

import lightning.pytorch as pl
import wandb.errors
from pytorch_lightning.loggers import WandbLogger

CIQUAL_KEYS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1012, 1013, 1014, 1015, 1017, 1018, 1019, 1021,
               1022, 1023, 1026, 2002, 2008, 2011, 2013, 2035, 2060, 2061, 2069, 2074, 2500, 3000, 3002, 4000, 4002,
               4003, 4004, 4008, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4026, 4027, 4028,
               4029, 4030, 4032, 4034, 4035, 4036, 4037, 4038, 4039, 4041, 4042, 4043, 4044, 4045, 4046, 4101, 4102,
               4103, 5000, 5001, 5002, 5004, 5005, 5006, 5007, 5008, 5010, 5011, 5020, 5021, 5022, 5030, 5100, 5200,
               5201, 5203, 5204, 5205, 5206, 5207, 5208, 5209, 5211, 6001, 6002, 6100, 6101, 6102, 6103, 6104, 6105,
               6110, 6111, 6112, 6122, 6123, 6130, 6131, 6140, 6141, 6150, 6151, 6160, 6161, 6162, 6200, 6201, 6202,
               6204, 6205, 6206, 6207, 6208, 6210, 6211, 6212, 6214, 6230, 6231, 6241, 6250, 6251, 6252, 6253, 6254,
               6255, 6256, 6257, 6260, 6270, 6271, 6310, 6510, 6511, 6512, 6513, 6520, 6521, 6522, 6523, 6524, 6530,
               6531, 6535, 6536, 6540, 6550, 6551, 6560, 6562, 6563, 6580, 6581, 6582, 6583, 6590, 6591, 7001, 7002,
               7004, 7007, 7010, 7012, 7025, 7100, 7110, 7111, 7112, 7113, 7115, 7125, 7130, 7160, 7170, 7180, 7200,
               7201, 7210, 7225, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7300, 7301, 7310, 7330, 7340, 7352,
               7353, 7400, 7403, 7407, 7409, 7410, 7412, 7413, 7420, 7421, 7425, 7430, 7432, 7525, 7602, 7615, 7620,
               7650, 7710, 7711, 7712, 7720, 7730, 7733, 7735, 7737, 7738, 7739, 7740, 7741, 7742, 7744, 7745, 7811,
               7812, 7813, 7814, 7815, 8000, 8001, 8010, 8015, 8025, 8026, 8030, 8040, 8080, 8081, 8082, 8083, 8109,
               8110, 8111, 8120, 8125, 8201, 8211, 8214, 8232, 8240, 8242, 8245, 8250, 8291, 8292, 8293, 8296, 8297,
               8300, 8305, 8312, 8313, 8315, 8316, 8326, 8373, 8391, 8395, 8400, 8406, 8500, 8501, 8504, 8512, 8550,
               8551, 8552, 8601, 8602, 8612, 8703, 8704, 8803, 8903, 8910, 8912, 8932, 8933, 8937, 9001, 9003, 9010,
               9060, 9080, 9081, 9082, 9083, 9085, 9086, 9100, 9101, 9102, 9103, 9104, 9105, 9108, 9109, 9110, 9111,
               9119, 9121, 9200, 9230, 9231, 9232, 9310, 9311, 9313, 9320, 9321, 9322, 9330, 9331, 9340, 9341, 9345,
               9360, 9380, 9390, 9410, 9415, 9435, 9436, 9437, 9440, 9445, 9480, 9510, 9520, 9530, 9532, 9533, 9540,
               9545, 9550, 9555, 9570, 9580, 9610, 9612, 9614, 9615, 9621, 9640, 9641, 9643, 9681, 9683, 9690, 9691,
               9810, 9811, 9815, 9816, 9821, 9822, 9824, 9863, 9870, 9871, 9874, 9900, 9901, 10001, 10002, 10003, 10004,
               10007, 10011, 10013, 10014, 10021, 10023, 10024, 10026, 10028, 10035, 10036, 10037, 10038, 10042, 10045,
               10048, 10049, 10059, 10081, 10082, 10083, 10084, 11000, 11002, 11003, 11005, 11006, 11007, 11008, 11009,
               11010, 11013, 11014, 11015, 11016, 11017, 11018, 11019, 11020, 11021, 11023, 11024, 11025, 11026, 11027,
               11032, 11033, 11034, 11035, 11036, 11037, 11038, 11039, 11042, 11043, 11044, 11045, 11046, 11048, 11049,
               11051, 11052, 11053, 11054, 11056, 11058, 11060, 11061, 11062, 11064, 11065, 11066, 11068, 11069, 11070,
               11074, 11075, 11077, 11079, 11082, 11083, 11084, 11086, 11088, 11089, 11090, 11091, 11092, 11093, 11094,
               11098, 11100, 11101, 11102, 11104, 11105, 11107, 11110, 11111, 11112, 11114, 11115, 11120, 11121, 11122,
               11128, 11129, 11132, 11140, 11143, 11157, 11158, 11159, 11160, 11161, 11162, 11163, 11164, 11166, 11167,
               11168, 11170, 11172, 11177, 11178, 11179, 11182, 11187, 11189, 11191, 11192, 11194, 11196, 11198, 11199,
               11202, 11203, 11205, 11207, 11208, 11210, 11212, 11214, 11300, 11507, 12001, 12003, 12006, 12008, 12009,
               12010, 12012, 12013, 12020, 12021, 12022, 12025, 12028, 12029, 12030, 12031, 12033, 12035, 12036, 12037,
               12038, 12039, 12040, 12042, 12045, 12047, 12049, 12050, 12051, 12052, 12060, 12061, 12063, 12105, 12110,
               12113, 12114, 12115, 12116, 12118, 12119, 12120, 12121, 12123, 12300, 12310, 12315, 12320, 12340, 12355,
               12356, 12500, 12519, 12520, 12521, 12522, 12523, 12524, 12526, 12527, 12528, 12705, 12716, 12722, 12723,
               12725, 12726, 12729, 12735, 12736, 12737, 12738, 12740, 12741, 12742, 12743, 12747, 12748, 12749, 12751,
               12752, 12755, 12758, 12759, 12760, 12761, 12762, 12763, 12800, 12801, 12802, 12803, 12804, 12805, 12807,
               12810, 12812, 12813, 12814, 12815, 12820, 12824, 12827, 12830, 12831, 12832, 12833, 12834, 12836, 12839,
               12842, 12845, 12846, 12847, 12848, 13000, 13001, 13002, 13004, 13005, 13007, 13008, 13009, 13010, 13011,
               13012, 13013, 13014, 13015, 13016, 13018, 13019, 13020, 13021, 13023, 13025, 13026, 13028, 13029, 13030,
               13034, 13035, 13036, 13037, 13038, 13039, 13040, 13041, 13042, 13043, 13044, 13045, 13046, 13047, 13048,
               13050, 13054, 13063, 13066, 13067, 13071, 13079, 13082, 13083, 13085, 13089, 13090, 13100, 13107, 13108,
               13109, 13110, 13111, 13112, 13113, 13118, 13125, 13126, 13129, 13132, 13134, 13136, 13147, 13150, 13152,
               13153, 13157, 13158, 13159, 13161, 13162, 13163, 13164, 13165, 13166, 13167, 13168, 13169, 13170, 13173,
               13175, 13176, 13179, 13180, 13549, 13614, 13620, 13706, 13707, 13708, 13709, 13712, 13713, 13714, 13715,
               13716, 13717, 13718, 13719, 13730, 13731, 13735, 13742, 13997, 15000, 15001, 15002, 15004, 15005, 15006,
               15007, 15008, 15009, 15010, 15011, 15013, 15014, 15016, 15018, 15019, 15020, 15021, 15023, 15024, 15025,
               15026, 15027, 15028, 15029, 15032, 15033, 15034, 15035, 15037, 15038, 15039, 15041, 15042, 15043, 15044,
               15045, 15046, 15047, 15048, 15049, 15050, 15052, 15201, 15202, 15203, 16030, 16080, 16128, 16129, 16150,
               16400, 16401, 16402, 16403, 16404, 16410, 16411, 16415, 16520, 16530, 16540, 16550, 16560, 16570, 16614,
               16615, 16616, 16654, 16712, 16713, 16733, 16734, 16735, 16736, 16737, 16738, 16739, 16740, 16741, 16742,
               16743, 16744, 16745, 16746, 17040, 17130, 17170, 17180, 17190, 17210, 17270, 17350, 17400, 17420, 17440,
               17630, 17640, 17645, 17650, 17700, 17701, 18003, 18004, 18005, 18010, 18012, 18013, 18014, 18015, 18016,
               18017, 18018, 18019, 18020, 18021, 18022, 18023, 18026, 18028, 18030, 18032, 18033, 18035, 18037, 18039,
               18041, 18049, 18058, 18060, 18065, 18067, 18068, 18069, 18070, 18071, 18072, 18073, 18075, 18078, 18100,
               18101, 18104, 18106, 18107, 18150, 18151, 18152, 18153, 18154, 18155, 18156, 18160, 18161, 18162, 18163,
               18167, 18168, 18220, 18304, 18309, 18339, 18340, 18343, 18344, 18345, 18352, 18353, 18430, 18900, 18901,
               18902, 18903, 18904, 18905, 19012, 19013, 19014, 19021, 19023, 19024, 19026, 19027, 19041, 19042, 19044,
               19049, 19050, 19051, 19054, 19060, 19122, 19127, 19200, 19201, 19202, 19250, 19410, 19415, 19420, 19430,
               19431, 19433, 19508, 19530, 19535, 19537, 19538, 19539, 19542, 19546, 19550, 19552, 19556, 19558, 19559,
               19575, 19577, 19579, 19580, 19581, 19582, 19587, 19589, 19590, 19592, 19593, 19594, 19598, 19599, 19641,
               19644, 19646, 19649, 19659, 19661, 19662, 19663, 19664, 19666, 19667, 19673, 19674, 19678, 19679, 19680,
               19681, 19683, 19685, 19689, 19692, 19693, 19698, 19801, 19805, 19852, 19860, 20000, 20001, 20002, 20003,
               20004, 20005, 20006, 20007, 20008, 20009, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20020,
               20021, 20022, 20023, 20024, 20025, 20026, 20027, 20028, 20029, 20030, 20031, 20033, 20034, 20035, 20036,
               20037, 20038, 20039, 20040, 20041, 20043, 20044, 20045, 20046, 20047, 20048, 20049, 20050, 20051, 20052,
               20053, 20054, 20055, 20057, 20058, 20059, 20060, 20061, 20062, 20063, 20064, 20065, 20066, 20067, 20068,
               20069, 20070, 20071, 20072, 20073, 20076, 20077, 20078, 20081, 20082, 20083, 20084, 20085, 20086, 20087,
               20088, 20089, 20090, 20091, 20093, 20094, 20095, 20096, 20097, 20099, 20101, 20105, 20108, 20111, 20116,
               20118, 20119, 20121, 20122, 20124, 20126, 20128, 20132, 20133, 20134, 20135, 20136, 20138, 20139, 20143,
               20145, 20151, 20155, 20156, 20158, 20162, 20163, 20165, 20166, 20167, 20168, 20169, 20170, 20171, 20172,
               20173, 20180, 20181, 20183, 20188, 20189, 20194, 20195, 20196, 20197, 20198, 20199, 20200, 20201, 20203,
               20204, 20205, 20206, 20207, 20208, 20209, 20210, 20214, 20215, 20216, 20217, 20218, 20219, 20221, 20230,
               20231, 20232, 20233, 20234, 20235, 20236, 20237, 20240, 20241, 20242, 20243, 20246, 20247, 20248, 20249,
               20250, 20251, 20252, 20253, 20254, 20255, 20256, 20257, 20258, 20259, 20260, 20261, 20262, 20263, 20264,
               20265, 20266, 20267, 20268, 20269, 20270, 20271, 20272, 20273, 20274, 20275, 20278, 20279, 20280, 20282,
               20283, 20284, 20285, 20289, 20496, 20497, 20498, 20500, 20501, 20502, 20503, 20504, 20505, 20506, 20507,
               20508, 20510, 20511, 20513, 20515, 20516, 20517, 20518, 20521, 20524, 20525, 20530, 20531, 20532, 20534,
               20535, 20536, 20537, 20539, 20540, 20541, 20542, 20543, 20581, 20585, 20586, 20587, 20588, 20589, 20900,
               20901, 20904, 20911, 20916, 20917, 20918, 20919, 20984, 20985, 20986, 20987, 20988, 20990, 20991, 20992,
               20993, 20994, 20995, 20996, 20998, 20999, 21001, 21003, 21004, 21005, 21006, 21500, 21501, 21502, 21503,
               21504, 21505, 21506, 21507, 21508, 21509, 21512, 21513, 21514, 21515, 21516, 21517, 21518, 21519, 21520,
               21800, 21801, 22000, 22001, 22002, 22003, 22004, 22008, 22009, 22010, 22011, 22013, 22014, 22060, 22070,
               22080, 22502, 22505, 22506, 22507, 22508, 22509, 22510, 23005, 23006, 23007, 23008, 23009, 23020, 23022,
               23024, 23027, 23030, 23032, 23033, 23050, 23081, 23103, 23121, 23122, 23200, 23300, 23301, 23402, 23403,
               23410, 23412, 23414, 23415, 23420, 23421, 23422, 23424, 23425, 23426, 23440, 23442, 23444, 23445, 23446,
               23448, 23455, 23456, 23457, 23467, 23472, 23474, 23477, 23479, 23480, 23481, 23485, 23490, 23491, 23493,
               23494, 23495, 23496, 23497, 23499, 23525, 23531, 23534, 23535, 23536, 23585, 23586, 23588, 23589, 23594,
               23680, 23684, 23799, 23800, 23801, 23802, 23803, 23805, 23815, 23820, 23821, 23829, 23830, 23851, 23852,
               23853, 23854, 23880, 23881, 23884, 23885, 23909, 23925, 23930, 23937, 23938, 23939, 23940, 23941, 23950,
               24000, 24001, 24002, 24003, 24004, 24007, 24008, 24009, 24010, 24011, 24015, 24016, 24017, 24030, 24031,
               24034, 24035, 24036, 24037, 24038, 24039, 24040, 24041, 24049, 24050, 24051, 24052, 24053, 24054, 24055,
               24056, 24060, 24070, 24071, 24072, 24080, 24225, 24231, 24240, 24300, 24311, 24312, 24313, 24320, 24360,
               24370, 24371, 24430, 24441, 24520, 24615, 24616, 24630, 24631, 24632, 24659, 24660, 24663, 24664, 24666,
               24678, 24679, 24680, 24684, 24685, 24686, 24689, 24690, 25001, 25002, 25003, 25004, 25009, 25010, 25013,
               25018, 25019, 25020, 25026, 25029, 25031, 25033, 25037, 25038, 25043, 25052, 25056, 25057, 25058, 25063,
               25065, 25071, 25073, 25077, 25081, 25085, 25086, 25088, 25089, 25098, 25099, 25101, 25103, 25106, 25107,
               25108, 25110, 25111, 25121, 25122, 25123, 25124, 25125, 25126, 25127, 25128, 25131, 25135, 25137, 25138,
               25139, 25140, 25141, 25142, 25143, 25145, 25146, 25149, 25150, 25151, 25152, 25154, 25155, 25157, 25158,
               25159, 25162, 25163, 25164, 25169, 25173, 25174, 25181, 25182, 25183, 25184, 25185, 25187, 25188, 25189,
               25190, 25192, 25193, 25195, 25196, 25198, 25199, 25203, 25204, 25207, 25208, 25211, 25213, 25218, 25219,
               25399, 25400, 25401, 25402, 25403, 25404, 25405, 25409, 25410, 25411, 25412, 25413, 25414, 25415, 25416,
               25417, 25418, 25419, 25420, 25428, 25429, 25431, 25433, 25434, 25435, 25437, 25438, 25444, 25454, 25456,
               25457, 25459, 25460, 25462, 25463, 25464, 25468, 25472, 25475, 25476, 25477, 25478, 25479, 25485, 25488,
               25490, 25502, 25503, 25504, 25505, 25506, 25508, 25509, 25510, 25511, 25512, 25513, 25517, 25518, 25519,
               25520, 25521, 25523, 25524, 25525, 25528, 25529, 25530, 25531, 25532, 25533, 25535, 25536, 25537, 25539,
               25541, 25542, 25544, 25546, 25547, 25548, 25549, 25550, 25551, 25552, 25553, 25555, 25556, 25557, 25558,
               25559, 25560, 25561, 25562, 25564, 25565, 25566, 25568, 25570, 25571, 25572, 25574, 25575, 25576, 25577,
               25581, 25582, 25583, 25584, 25587, 25588, 25600, 25601, 25602, 25604, 25605, 25606, 25608, 25609, 25610,
               25614, 25615, 25619, 25620, 25621, 25623, 25625, 25628, 25635, 25900, 25901, 25903, 25904, 25905, 25907,
               25908, 25910, 25912, 25913, 25914, 25915, 25916, 25917, 25919, 25923, 25924, 25925, 25928, 25932, 25933,
               25934, 25935, 25936, 25942, 25945, 25948, 25949, 25950, 25953, 25954, 25955, 25956, 25957, 25958, 25962,
               25963, 25964, 25965, 25967, 25968, 25998, 25999, 26000, 26001, 26002, 26003, 26006, 26008, 26009, 26010,
               26011, 26012, 26013, 26014, 26015, 26016, 26017, 26018, 26019, 26020, 26021, 26022, 26023, 26024, 26025,
               26026, 26027, 26028, 26029, 26030, 26031, 26033, 26034, 26035, 26036, 26037, 26038, 26039, 26040, 26041,
               26042, 26043, 26044, 26046, 26047, 26048, 26051, 26052, 26053, 26054, 26057, 26058, 26059, 26060, 26061,
               26062, 26063, 26064, 26065, 26068, 26071, 26072, 26073, 26074, 26075, 26076, 26077, 26079, 26080, 26081,
               26082, 26083, 26084, 26085, 26086, 26087, 26088, 26090, 26091, 26092, 26093, 26094, 26095, 26096, 26097,
               26098, 26099, 26100, 26101, 26102, 26103, 26104, 26106, 26107, 26108, 26109, 26110, 26111, 26113, 26119,
               26120, 26122, 26123, 26124, 26126, 26127, 26128, 26129, 26130, 26133, 26134, 26135, 26136, 26146, 26147,
               26148, 26152, 26153, 26154, 26157, 26159, 26162, 26166, 26170, 26171, 26172, 26173, 26174, 26177, 26178,
               26179, 26181, 26186, 26187, 26194, 26200, 26201, 26204, 26205, 26206, 26210, 26211, 26213, 26214, 26217,
               26219, 26222, 26229, 26230, 26231, 26232, 26233, 26234, 26235, 26236, 26237, 26238, 26239, 26240, 26241,
               26242, 26243, 26244, 26245, 26247, 26248, 26256, 26259, 26264, 26265, 26266, 26267, 26268, 26269, 26270,
               26271, 26272, 26273, 26274, 27003, 27004, 27005, 27006, 27007, 27008, 27009, 27010, 27014, 27015, 27017,
               27018, 27021, 27025, 27029, 27030, 28001, 28002, 28003, 28004, 28007, 28010, 28100, 28101, 28102, 28103,
               28104, 28105, 28201, 28202, 28203, 28204, 28300, 28301, 28302, 28400, 28401, 28451, 28460, 28461, 28470,
               28471, 28472, 28473, 28474, 28475, 28476, 28477, 28478, 28479, 28480, 28501, 28502, 28503, 28504, 28530,
               28540, 28550, 28700, 28720, 28725, 28727, 28800, 28801, 28802, 28803, 28804, 28811, 28812, 28844, 28845,
               28850, 28858, 28900, 28901, 28902, 28905, 28906, 28907, 28910, 28911, 28912, 28913, 28917, 28922, 28924,
               28925, 28927, 28929, 28960, 28963, 28964, 28976, 30005, 30011, 30050, 30051, 30052, 30104, 30105, 30108,
               30110, 30115, 30118, 30125, 30130, 30131, 30134, 30150, 30152, 30153, 30154, 30155, 30156, 30176, 30177,
               30300, 30301, 30302, 30304, 30309, 30311, 30315, 30316, 30317, 30350, 30351, 30352, 30366, 30700, 30701,
               30705, 30706, 30707, 30730, 30731, 30732, 30742, 30746, 30750, 30764, 30766, 30789, 30790, 30791, 30797,
               31000, 31001, 31002, 31003, 31004, 31005, 31008, 31009, 31010, 31012, 31013, 31014, 31016, 31017, 31018,
               31020, 31024, 31026, 31030, 31032, 31033, 31035, 31036, 31037, 31038, 31039, 31040, 31041, 31042, 31044,
               31050, 31059, 31060, 31062, 31063, 31064, 31066, 31067, 31069, 31070, 31071, 31072, 31073, 31074, 31077,
               31079, 31080, 31081, 31084, 31085, 31091, 31092, 31098, 31099, 31100, 31101, 31102, 31106, 31110, 31113,
               31114, 32000, 32001, 32002, 32005, 32008, 32009, 32011, 32013, 32014, 32016, 32017, 32018, 32021, 32022,
               32023, 32025, 32028, 32029, 32030, 32107, 32108, 32109, 32110, 32111, 32112, 32113, 32115, 32116, 32121,
               32123, 32128, 32129, 32133, 32134, 32135, 32138, 32140, 34000, 34001, 34002, 34003, 34004, 36000, 36001,
               36002, 36003, 36004, 36005, 36006, 36007, 36008, 36014, 36016, 36017, 36018, 36019, 36020, 36022, 36023,
               36024, 36027, 36029, 36030, 36031, 36032, 36033, 36035, 36036, 36050, 36051, 36102, 36200, 36201, 36202,
               36203, 36204, 36205, 36206, 36300, 36301, 36302, 36304, 36305, 36306, 36307, 36308, 36310, 36318, 36402,
               36500, 36501, 36502, 36503, 36602, 36700, 36702, 36703, 37000, 37001, 37002, 38104, 38105, 38106, 38107,
               38108, 38400, 38401, 38402, 38403, 38404, 38405, 38406, 38407, 38408, 38500, 39001, 39200, 39206, 39209,
               39210, 39211, 39212, 39213, 39214, 39215, 39216, 39218, 39220, 39228, 39229, 39232, 39234, 39235, 39236,
               39246, 39247, 39401, 39502, 39503, 39506, 39509, 39512, 39515, 39516, 39517, 39518, 39519, 39520, 39521,
               39522, 39523, 39524, 39526, 39529, 39530, 39531, 39532, 39533, 39534, 39700, 39710, 40002, 40003, 40004,
               40005, 40006, 40007, 40052, 40053, 40054, 40055, 40056, 40057, 40058, 40059, 40060, 40062, 40102, 40103,
               40104, 40105, 40106, 40107, 40108, 40110, 40111, 40113, 40115, 40116, 40118, 40119, 40120, 40121, 40201,
               40202, 40203, 40204, 40205, 40302, 40303, 40304, 40305, 40402, 40403, 40404, 40405, 40406, 40407, 40408,
               40409, 40502, 40600, 40700, 42000, 42003, 42004, 42005, 42200, 42603, 42604, 42605, 42606, 51500, 51510,
               51511, 51550, 53100, 53101, 53200, 53201, 53502, 53503, 54031, 54034, 54500, 58103, 76000, 76001, 76002,
               76004, 76006, 76007, 76008, 76010, 76011, 76012, 76013, 76014, 76015, 76016, 76017, 76018, 76019, 76020,
               76022, 76023, 76024, 76025, 76027, 76028, 76029, 76030, 76031, 76032, 76033, 76034, 76035, 76036, 76037,
               76038, 76039, 76040, 76043, 76044, 76046, 76047, 76049, 76050, 76053, 76054, 76055, 76056, 76057, 76058,
               76059, 76060, 76061, 76062, 76063, 76064, 76065, 76066, 76067, 76068, 76069, 76070, 76071, 76072, 76073,
               76074, 76075, 76076, 76077, 76078, 76079, 76080, 76081, 76082, 76083, 76085, 76086, 76087, 76088, 76089,
               76090, 76091, 76092, 76093, 76094, 76095, 76096, 76097, 76100, 76101, 76102, 96778]
IDX_TO_CIQUAL = {idx: k for idx, k in enumerate(CIQUAL_KEYS)}
CIQUAL_TO_IDX = {k: idx for idx, k in enumerate(CIQUAL_KEYS)}


def get_callbacks(cfg):
    key = "val_loss/dataloader_idx_0"
    return [
        pl.callbacks.ModelCheckpoint(
            monitor=key,
            save_on_train_epoch_end=False,
            dirpath=cfg.save_dir,
            every_n_epochs=1
        ),
        pl.callbacks.EarlyStopping(
            monitor=key,
            min_delta=cfg.es_delta,
            patience=cfg.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    ]


def get_wandb_logger(cfg):
    try:
        wandb_logger = WandbLogger(project="harrygobert")
    except wandb.errors.UsageError:
        from getpass import getpass
        wandb.login(key=getpass("wandb API token:"))
        wandb_logger = WandbLogger(project="harrygobert")
    return wandb_logger


def parse_args():
    parser = argparse.ArgumentParser()
    root_path = os.path.abspath(os.path.join(__file__, "../.."))

    # Training settings
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    parser.add_argument('--model_name', default="distilbert-base-multilingual-cased", type=str,
                        help='Name of the pre-trained model')
    parser.add_argument('--n_accumulation_steps', default=1, type=int, help='Number of steps to accumulate gradients')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Ratio of steps for warmup phase')
    parser.add_argument('--max_len', default=32, type=int, help='Maximum sequence length')
    parser.add_argument('--num_steps', default=8000, type=int, help='Number of steps to train for')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate for optimizer')
    parser.add_argument('--llrd', default=0.7, type=float, help='Layer-wise learning rate decay')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay')
    parser.add_argument('--eval_steps', default=100, type=int, help='After how many steps to do evaluation')
    parser.add_argument('--grid_search', default=False, type=bool, help='Whether to run grid search')
    parser.add_argument('--n_folds', default=1, type=int,
                        help='Number of cross-validation folds. 0 trains on full data.')

    # Artefact settings
    parser.add_argument('--save_dir', default=os.path.join(root_path, 'model'), type=str,
                        help='Path to save trained model to')
    parser.add_argument('--quantize', default=False, type=bool, help='Whether or not to quantize the output model')
    parser.add_argument('--es_delta', default=0.01, type=float, help='Early stopping delta')
    parser.add_argument('--es_patience', default=10, type=int, help='Early stopping patience')

    # Data settings
    parser.add_argument('--translate', default=True, type=bool, help='Whether to translate text')
    parser.add_argument('--use_cached', default=False, type=bool, help='Whether to use cached data')
    parser.add_argument('--use_subcats', default=False, type=bool, help='Whether to use sub-categories')
    parser.add_argument('--n_classes', default=2473, type=int, help='Number of classes')
    parser.add_argument('--agribalyse_path',
                        default=os.path.join(root_path, 'data/product_to_ciqual.yaml'), type=str,
                        help='Path to Agribalyse data')
    parser.add_argument('--ciqual_dict', default=os.path.join(root_path, 'data/ciqual_dict.yaml'),
                        type=str, help='Path to full CIQUAL data')
    parser.add_argument('--ciqual_to_name_path', default=os.path.join(root_path, 'data/ciqual_to_lci_name.yaml'),
                        type=str, help='Path to CIQUAL name dict')
    parser.add_argument('--csv_path', default=os.path.join(root_path, 'data/products.csv'), type=str,
                        help='Path to CSV products data')
    parser.add_argument('--cache_path', default=os.path.join(root_path, 'data/cache'), type=str,
                        help='Path to CSV products data')

    # Logging settings
    parser.add_argument('--run_name', default="HGV-debug", type=str, help='Name of the run')
    parser.add_argument('--use_wandb', default=True, type=bool, help='Whether to use wandb')
    return parser.parse_args()
