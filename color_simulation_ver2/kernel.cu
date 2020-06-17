#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>

#define D65_ROW 531		// D65の行数
#define D65_COL 2		// D65の列数
#define OBS_ROW 441		// 標準観測者の行数
#define OBS_COL 4		// 標準観測者の列数
#define XYZ_ROW 471		// xyzの行数
#define XYZ_COL 4		// xyzの列数
#define DATA_ROW 441	// 計算で使用するデータの行数 (390 - 830 nm)
#define DATA_MIN 390	// 使用する周波数の最小値
#define DATA_MAX 830	// 使用する周波数の最大値
#define PI 3.141592		// 円周率

#define BLOCKSIZE 441		// 1ブロック当たりのスレッド数
#define DATANUM 50			// 計算する数
#define CALCNUM 25000		// べき乗する数
#define SIMNUM 1023			// シミュレーションする回数
#define LOOPNUM 10			// SIMNUM回のシミュレーション繰り返す回数

using namespace std;