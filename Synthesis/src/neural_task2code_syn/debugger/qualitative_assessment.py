import heapq
import random
from os.path import relpath

import torch
from src.agent.debugger_agent import beam_search
from src.agent.debugger_agent.sed_code import arguments
from src.agent.debugger_agent.sed_code.arguments import backport_default_args
from src.agent.debugger_agent.sed_code.data import PlaceholderVocab, load_vocab
from src.agent.debugger_agent.sed_code.executor import KarelExecutor, evaluate_code
from src.agent.debugger_agent.sed_code.karel_model import KarelLGRLRefineBatchProcessor, \
    KarelLGRLRefineModel, maybe_cuda
from src.agent.debugger_agent.sed_code.saver import restore_args

from src.karel_emulator.world import World

# tgt_code = ['DEF', 'run', 'm(', 'move', 'move', 'REPEAT', 'R=5', 'r(', 'turnLeft', 'r)',
#             'putMarker', 'move', 'putMarker', 'putMarker', 'move', 'putMarker', 'WHILE',
#             'c(', 'leftIsClear', 'c)', 'w(', 'move', 'w)', 'm)']
# wrong_code1 = (
#     'DEF', 'run', 'm(', 'move', 'move', 'turnLeft', 'WHILE', 'c(', 'leftIsClear', 'c)',
#     'w(', 'putMarker', 'move', 'putMarker', 'w)', 'move', 'm)')
# wrong_code2 = ('DEF', 'run', 'm(', 'move', 'move', 'WHILE', 'c(', 'leftIsClear', 'c)',
#                'w(', 'putMarker', 'move', 'putMarker', 'w)', 'move', 'm)')
# wrong_code3 = ('DEF', 'run', 'm(', 'move', 'move', 'putMarker', 'move', 'putMarker',
#                'move', 'm)')
# wrong_code4 = ('DEF', 'run', 'm(', 'move', 'move', 'move', 'putMarker', 'move', 'm)')
# wrong_code5 = ('DEF', 'run', 'm(', 'move', 'move', 'putMarker', 'move', 'm)')
# wrong_code = ('DEF', 'run', 'm(', 'move', 'move', 'REPEAT', 'R=5', 'r(', 'turnLeft',
#               'r)',
#               'putMarker', 'move', 'putMarker', 'putMarker', 'putMarker', 'WHILE',
#               'c(', 'leftIsClear', 'c)', 'w(', 'move', 'w)', 'm)')
#
# example = {'input': [1084, 1317, 1319, 1333, 1334, 1335, 1336, 1337, 1353, 1371,
# 1372, 1387, 1389, 1391, 1409, 1423, 1424, 1427, 1441, 1442, 1445, 1463, 1479, 1620,
# 1621, 1622, 1623, 1624, 1625, 1626, 1638, 1644, 1656, 1662, 1674, 1680, 1692, 1698,
# 1710, 1716, 1728, 1734, 1746, 1752, 1764, 1770, 1782, 1788, 1800, 1806, 1818, 1819,
# 1820, 1821, 1822, 1823, 1824], 'output': [722, 1317, 1319, 1333, 1334, 1335, 1336,
# 1337, 1353, 1371, 1372, 1387, 1389, 1391, 1409, 1423, 1424, 1427, 1441, 1442, 1445,
# 1463, 1479, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1638, 1644, 1656, 1662, 1674,
# 1680, 1692, 1698, 1710, 1716, 1728, 1734, 1746, 1752, 1764, 1770, 1782, 1788, 1800,
# 1806, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 2018, 2054, 2360]}

# tgt_code = ['DEF', 'run', 'm(', 'REPEAT', 'R=5', 'r(', 'REPEAT', 'R=3', 'r(', 'IF', 'c(', 'markersPresent', 'c)', 'i(', 'pickMarker', 'i)', 'r)', 'move', 'r)', 'm)']
# wrong_code = ('DEF', 'run', 'm(', 'REPEAT', 'R=5', 'r(', 'IF', 'c(', 'markersPresent', 'c)', 'i(', 'pickMarker', 'i)', 'move', 'r)', 'm)')
#
# example = [{'input': [85, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1638, 1654, 1656, 1672, 1674, 1690, 1692, 1708, 1710, 1726, 1728, 1744, 1746, 1762, 1764, 1780, 1782, 1798, 1800, 1816, 1818, 1834, 1836, 1852, 1854, 1870, 1872, 1888, 1890, 1906, 1908, 1924, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1963, 1968, 1971, 1991, 2001, 2018, 2021, 2027, 2029, 2053, 2075, 2076, 2081, 2098, 2120, 2147, 2149, 2156, 2162, 2166, 2174, 2186, 2192, 2207, 2225, 2229, 2235, 2236, 2237, 2297, 2442, 2532, 2671, 3478], 'output': [175, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1638, 1654, 1656, 1672, 1674, 1690, 1692, 1708, 1710, 1726, 1728, 1744, 1746, 1762, 1764, 1780, 1782, 1798, 1800, 1816, 1818, 1834, 1836, 1852, 1854, 1870, 1872, 1888, 1890, 1906, 1908, 1924, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1963, 1968, 1971, 1991, 2001, 2018, 2021, 2027, 2053, 2075, 2076, 2081, 2098, 2120, 2147, 2149, 2156, 2162, 2166, 2174, 2186, 2192, 2207, 2225, 2229, 2235, 2236, 2237, 2297, 2442, 2532, 2671, 3478]}, {'input': [344, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1638, 1646, 1656, 1664, 1674, 1682, 1692, 1700, 1710, 1718, 1728, 1736, 1746, 1754, 1764, 1772, 1782, 1790, 1800, 1808, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1965, 1982, 2018, 2055, 2111, 2127, 2130, 2378, 3714, 3983], 'output': [349, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1638, 1646, 1656, 1664, 1674, 1682, 1692, 1700, 1710, 1718, 1728, 1736, 1746, 1754, 1764, 1772, 1782, 1790, 1800, 1808, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1982, 2018, 2055, 2111, 2127, 2130, 2378, 3714, 3983]}, {'input': [19, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1638, 1649, 1656, 1667, 1674, 1685, 1692, 1703, 1710, 1721, 1728, 1739, 1746, 1757, 1764, 1775, 1782, 1793, 1800, 1811, 1818, 1829, 1836, 1847, 1854, 1865, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1987, 2004, 2025, 2035, 2037, 2071, 2114, 2130, 2162, 2169, 2509, 3295], 'output': [109, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1638, 1649, 1656, 1667, 1674, 1685, 1692, 1703, 1710, 1721, 1728, 1739, 1746, 1757, 1764, 1775, 1782, 1793, 1800, 1811, 1818, 1829, 1836, 1847, 1854, 1865, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1987, 2004, 2025, 2037, 2071, 2114, 2130, 2162, 2169, 2323, 2509]}, {'input': [50, 1327, 1333, 1340, 1351, 1360, 1362, 1371, 1376, 1390, 1401, 1409, 1412, 1413, 1426, 1434, 1442, 1452, 1466, 1483, 1484, 1509, 1518, 1522, 1526, 1531, 1538, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1655, 1656, 1673, 1674, 1691, 1692, 1709, 1710, 1727, 1728, 1745, 1746, 1763, 1764, 1781, 1782, 1799, 1800, 1817, 1818, 1835, 1836, 1853, 1854, 1871, 1872, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 2012, 2020, 2026, 2043, 2048, 2071, 2104, 2118, 2140, 2145, 2150, 2203, 2210, 2317, 2985, 3337, 3362, 3612, 3809, 3929], 'output': [140, 1327, 1333, 1340, 1351, 1360, 1362, 1371, 1376, 1390, 1401, 1409, 1412, 1413, 1426, 1434, 1442, 1452, 1466, 1483, 1484, 1509, 1518, 1522, 1526, 1531, 1538, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1655, 1656, 1673, 1674, 1691, 1692, 1709, 1710, 1727, 1728, 1745, 1746, 1763, 1764, 1781, 1782, 1799, 1800, 1817, 1818, 1835, 1836, 1853, 1854, 1871, 1872, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 2020, 2026, 2043, 2071, 2104, 2118, 2140, 2145, 2150, 2203, 2210, 2317, 2390, 2985, 3337, 3612, 3809, 3929]}, {'input': [883, 1620, 1621, 1622, 1623, 1624, 1638, 1642, 1656, 1660, 1674, 1678, 1692, 1696, 1710, 1714, 1728, 1732, 1746, 1750, 1764, 1768, 1782, 1786, 1800, 1804, 1818, 1822, 1836, 1840, 1854, 1858, 1872, 1876, 1890, 1894, 1908, 1912, 1926, 1927, 1928, 1929, 1930, 2109, 2216, 2217, 3007, 3727], 'output': [793, 1620, 1621, 1622, 1623, 1624, 1638, 1642, 1656, 1660, 1674, 1678, 1692, 1696, 1710, 1714, 1728, 1732, 1746, 1750, 1764, 1768, 1782, 1786, 1800, 1804, 1818, 1822, 1836, 1840, 1854, 1858, 1872, 1876, 1890, 1894, 1908, 1912, 1926, 1927, 1928, 1929, 1930, 2109, 2216, 2217, 2755, 3007]}]

tgt_code = ['DEF', 'run', 'm(', 'putMarker', 'move', 'putMarker', 'REPEAT', 'R=7', 'r(',
            'move', 'WHILE', 'c(', 'noMarkersPresent', 'c)', 'w(', 'putMarker', 'w)',
            'r)', 'turnLeft', 'm)']
wrong_code = (
'DEF', 'run', 'm(', 'putMarker', 'REPEAT', 'R=4', 'r(', 'move', 'putMarker', 'move',
'IFELSE', 'c(', 'markersPresent', 'c)', 'i(', 'move', 'i)', 'ELSE', 'e(', 'putMarker',
'e)', 'r)', 'turnLeft', 'm)')

example = [{'input': [829, 1620, 1621, 1622, 1623, 1638, 1641, 1656, 1659, 1674, 1677,
                      1692, 1695, 1710, 1713, 1728, 1731, 1746, 1749, 1764, 1767, 1782,
                      1785, 1800, 1803, 1818, 1821, 1836, 1839, 1854, 1855, 1856, 1857,
                      1964, 2072],
            'output': [361, 1620, 1621, 1622, 1623, 1638, 1641, 1656, 1659, 1674, 1677,
                       1692, 1695, 1710, 1713, 1728, 1731, 1746, 1749, 1764, 1767, 1782,
                       1785, 1800, 1803, 1818, 1821, 1836, 1839, 1854, 1855, 1856, 1857,
                       1964, 1981, 1999, 2017, 2035, 2053, 2071, 2072, 2089, 2107,
                       2125]}, {
               'input': [19, 1334, 1337, 1373, 1620, 1621, 1622, 1623, 1624, 1625, 1626,
                         1627, 1638, 1645, 1656, 1663, 1674, 1681, 1692, 1699, 1710,
                         1717, 1728, 1735, 1746, 1753, 1764, 1771, 1782, 1789, 1800,
                         1801, 1802, 1803, 1804, 1805, 1806, 1807, 1963, 3587],
               'output': [1135, 1334, 1337, 1373, 1620, 1621, 1622, 1623, 1624, 1625,
                          1626, 1627, 1638, 1645, 1656, 1663, 1674, 1681, 1692, 1699,
                          1710, 1717, 1728, 1735, 1746, 1753, 1764, 1771, 1782, 1789,
                          1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1981, 1999,
                          2017, 2035, 2053, 2071, 2089, 2107, 2287, 3587]}, {
               'input': [343, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                         1629, 1630, 1631, 1632, 1638, 1650, 1656, 1668, 1674, 1686,
                         1692, 1704, 1710, 1722, 1728, 1740, 1746, 1758, 1764, 1776,
                         1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791,
                         1792, 1793, 1794, 1967, 1968, 1991, 1999, 2095, 2099, 3713,
                         3917, 4651],
               'output': [27, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                          1629, 1630, 1631, 1632, 1638, 1650, 1656, 1668, 1674, 1686,
                          1692, 1704, 1710, 1722, 1728, 1740, 1746, 1758, 1764, 1776,
                          1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791,
                          1792, 1793, 1794, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
                          1970, 1971, 1991, 1999, 2095, 2099, 3713, 3917, 4651]}, {
               'input': [418, 1381, 1389, 1399, 1407, 1452, 1477, 1482, 1505, 1513,
                         1519, 1520, 1523, 1536, 1543, 1556, 1620, 1621, 1622, 1623,
                         1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633,
                         1634, 1635, 1638, 1653, 1656, 1671, 1674, 1689, 1692, 1707,
                         1710, 1725, 1728, 1743, 1746, 1761, 1764, 1779, 1782, 1797,
                         1800, 1815, 1818, 1833, 1836, 1851, 1854, 1869, 1872, 1887,
                         1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
                         1900, 1901, 1902, 1903, 1904, 1905, 1968, 1992, 2000, 2039,
                         2057, 2059, 2061, 2071, 2089, 2091, 2094, 2155, 2182, 2198,
                         2201, 3326, 3595, 4133],
               'output': [102, 1381, 1389, 1399, 1407, 1452, 1477, 1482, 1505, 1513,
                          1519, 1520, 1523, 1536, 1543, 1556, 1620, 1621, 1622, 1623,
                          1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633,
                          1634, 1635, 1638, 1653, 1656, 1671, 1674, 1689, 1692, 1707,
                          1710, 1725, 1728, 1743, 1746, 1761, 1764, 1779, 1782, 1797,
                          1800, 1815, 1818, 1833, 1836, 1851, 1854, 1869, 1872, 1887,
                          1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
                          1900, 1901, 1902, 1903, 1904, 1905, 1968, 1992, 2000, 2038,
                          2040, 2041, 2042, 2043, 2044, 2045, 2046, 2057, 2059, 2061,
                          2071, 2089, 2091, 2094, 2155, 2182, 2198, 2201, 2363, 3326,
                          3595, 4133]}, {
               'input': [110, 1315, 1322, 1324, 1333, 1339, 1393, 1432, 1481, 1482,
                         1487, 1502, 1558, 1577, 1620, 1621, 1622, 1623, 1624, 1625,
                         1626, 1627, 1628, 1629, 1630, 1631, 1632, 1638, 1650, 1656,
                         1668, 1674, 1686, 1692, 1704, 1710, 1722, 1728, 1740, 1746,
                         1758, 1764, 1776, 1782, 1794, 1800, 1812, 1818, 1830, 1836,
                         1848, 1854, 1866, 1872, 1884, 1890, 1902, 1908, 1909, 1910,
                         1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920],
               'output': [1226, 1315, 1322, 1324, 1333, 1339, 1393, 1432, 1481, 1482,
                          1487, 1502, 1558, 1577, 1620, 1621, 1622, 1623, 1624, 1625,
                          1626, 1627, 1628, 1629, 1630, 1631, 1632, 1638, 1650, 1656,
                          1668, 1674, 1686, 1692, 1704, 1710, 1722, 1728, 1740, 1746,
                          1758, 1764, 1776, 1782, 1794, 1800, 1812, 1818, 1830, 1836,
                          1848, 1854, 1866, 1872, 1884, 1890, 1902, 1908, 1909, 1910,
                          1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920,
                          2054, 2072, 2090, 2108, 2126, 2144, 2162, 2180, 2198]}]


def get_tensors(example):
    input_grids, output_grids = [
        torch.zeros(16, 18, 18) for _ in range(2)
    ]
    inp, out = example['input'], example['output']
    input_grids.view(-1)[inp] = 1
    output_grids.view(-1)[out] = 1
    return input_grids, output_grids


if __name__ == '__main__':

    input_grids, output_grids = get_tensors(example[0])

    i = World.fromPytorchTensor(input_grids)
    o = World.fromPytorchTensor(output_grids)

    print(i.toString())
    print()
    print(o.toString())

    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')
    args = parser.parse_args()

    args.word_vocab = relpath('sed_code/word.vocab')
    vocab_sed = load_vocab(args.word_vocab)
    rev_vocab = {idx: key for key, idx in vocab_sed.items()}

    aux_vocab = PlaceholderVocab(
        vocab_sed, args.num_placeholders)
    dev_batch_processor = KarelLGRLRefineBatchProcessor(args, aux_vocab, True)

    restore_args(args)
    backport_default_args(args)

    debugger = KarelLGRLRefineModel(args)
    debugger.model.eval()

    executor = KarelExecutor()

    stats = evaluate_code(wrong_code, None, example, executor.execute)
    max_code = wrong_code
    max_score = stats['correct']
    print(max_score)
    for i in range(100):

        ios = {'in_grids': [], 'out_grids': []}
        for ex in example:
            i, o = get_tensors(ex)
            ios['in_grids'].append(i)
            ios['out_grids'].append(o)
        ios = {k: torch.stack(v) for k, v in ios.items()}
        batch = dev_batch_processor([ios],
                                    tuple(max_code),
                                    tgt_code)

        input_grids, output_grids, _1, dec_data, ref_code, \
        ref_trace_grids, ref_trace_events, cag_interleave, _2 = batch
        if True:
            input_grids = input_grids.cuda(non_blocking=True)
            output_grids = output_grids.cuda(non_blocking=True)
            dec_data = maybe_cuda(dec_data, non_blocking=True)
            ref_code = maybe_cuda(ref_code, non_blocking=True)
            ref_trace_grids = maybe_cuda(ref_trace_grids, non_blocking=True)
            ref_trace_events = maybe_cuda(ref_trace_events,
                                          non_blocking=True)

        io_embed, ref_code_memory, ref_trace_memory = debugger.model.encode(
            input_grids, output_grids, ref_code, ref_trace_grids,
            ref_trace_events, cag_interleave)
        init_state = debugger.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = debugger.model.decoder.prepare_memory(io_embed,
                                                       ref_code_memory,
                                                       ref_trace_memory,
                                                       ref_code)

        pre_paths = beam_search.beam_search(len(input_grids),
                                            init_state,
                                            memory,
                                            debugger.model.decode_token,
                                            32,
                                            cuda=True,
                                            max_decoder_length=100,
                                            return_attention=False,
                                            return_beam_search_result=False,
                                            differentiable=False,
                                            use_length_penalty=False,
                                            factor=0.7)

        pre_paths = heapq.nlargest(32, pre_paths[0],
                                   key=lambda x: x[0])
        pre_paths = [[y[1] for y in pre_paths]]
        code = [debugger.model.decoder.postprocess_output(pre_paths,
                                                          memory)]
        code = [[rev_vocab[x] for x in y] for y in code[0][0]]
        for line in code:
            stats = evaluate_code(line, None, example, executor.execute)
            if stats['correct'] > max_score:
                max_code = line
                max_score = stats['correct']
            elif stats['correct'] == max_score:
                if random.random() < 0.75:
                    max_code = line
                    max_score = stats['correct']
    print(max_code)
    print(max_score)

# ['DEF', 'run', 'm(', 'putMarker', 'REPEAT', 'R=4', 'r(', 'move', 'putMarker', 'move', 'putMarker', 'r)', 'turnLeft', 'm)']
