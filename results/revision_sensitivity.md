# Sensitivity of the paper-sample Sharpe ratios to small changes in the data

Independent normal noise is added to every monthly return of a dataset and the rolling test is rerun, 100 times for each noise size. A cell shows the Sharpe ratio on the files as downloaded, then the 5th to 95th percentile over the noisy copies in brackets, then the paper's figure. A star marks a paper figure outside the full range of the noisy copies. One basis point is the precision the library publishes to. The larger sizes are assumptions about how large revisions might be. They were not measured.

## Noise with a standard deviation of 1 basis points a month

| rule | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|
| ew | 0.1353 [0.1352, 0.1354] paper 0.1353 | 0.2347 [0.2345, 0.2350] paper 0.2240* | 0.1613 [0.1613, 0.1613] paper 0.1623* | 0.1745 [0.1744, 0.1745] paper 0.1753* |
| mv-is | 0.2034 [0.2026, 0.2042] paper 0.2124* | 0.2608 [0.2605, 0.2610] paper 0.2851* | 0.4509 [0.4505, 0.4514] paper 0.5098* | 0.4999 [0.4994, 0.5006] paper 0.5364* |
| mv | -0.0494 [-0.0530, -0.0378] paper 0.0679* | 0.2135 [0.2131, 0.2139] paper 0.2186* | 0.0513 [-0.0202, 0.0666] paper 0.0128 | 0.1650 [0.0139, 0.1773] paper 0.1841 |
| min | 0.1537 [0.1528, 0.1545] paper 0.1554* | 0.2574 [0.2571, 0.2578] paper 0.2493* | 0.2496 [0.2490, 0.2503] paper 0.2778* | 0.0326 [0.0296, 0.0357] paper -0.0183* |

## Noise with a standard deviation of 5 basis points a month

| rule | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|
| ew | 0.1353 [0.1351, 0.1356] paper 0.1353 | 0.2347 [0.2334, 0.2362] paper 0.2240* | 0.1613 [0.1612, 0.1615] paper 0.1623* | 0.1745 [0.1743, 0.1747] paper 0.1753* |
| mv-is | 0.2034 [0.1995, 0.2078] paper 0.2124* | 0.2608 [0.2593, 0.2621] paper 0.2851* | 0.4509 [0.4489, 0.4533] paper 0.5098* | 0.4999 [0.4970, 0.5026] paper 0.5364* |
| mv | -0.0494 [-0.0594, 0.0156] paper 0.0679* | 0.2135 [0.2115, 0.2155] paper 0.2186* | 0.0513 [-0.0358, 0.0647] paper 0.0128 | 0.1650 [-0.0122, 0.2105] paper 0.1841 |
| min | 0.1537 [0.1496, 0.1570] paper 0.1554 | 0.2574 [0.2560, 0.2592] paper 0.2493* | 0.2496 [0.2464, 0.2532] paper 0.2778* | 0.0326 [0.0153, 0.0526] paper -0.0183* |

## Noise with a standard deviation of 20 basis points a month

| rule | Industry | MKT/SMB/HML | FF-1-factor | FF-4-factor |
|---|---|---|---|---|
| ew | 0.1353 [0.1340, 0.1366] paper 0.1353 | 0.2347 [0.2296, 0.2392] paper 0.2240* | 0.1613 [0.1605, 0.1620] paper 0.1623 | 0.1745 [0.1737, 0.1752] paper 0.1753 |
| mv-is | 0.2034 [0.1858, 0.2163] paper 0.2124 | 0.2608 [0.2539, 0.2648] paper 0.2851* | 0.4509 [0.4410, 0.4576] paper 0.5098* | 0.4999 [0.4918, 0.5100] paper 0.5364* |
| mv | -0.0494 [-0.0697, 0.0569] paper 0.0679* | 0.2135 [0.2045, 0.2206] paper 0.2186 | 0.0513 [-0.0494, 0.0819] paper 0.0128 | 0.1650 [-0.0133, 0.2140] paper 0.1841 |
| min | 0.1537 [0.1408, 0.1600] paper 0.1554 | 0.2574 [0.2502, 0.2631] paper 0.2493 | 0.2496 [0.2321, 0.2622] paper 0.2778* | 0.0326 [0.0038, 0.0985] paper -0.0183 |
