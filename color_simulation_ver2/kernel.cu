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
#include <random>

#define D65_ROW 531		// D65の行数
#define D65_COL 2		// D65の列数
#define OBS_ROW 441		// 標準観測者の行数
#define OBS_COL 4		// 標準観測者の列数
#define XYZ_ROW 471		// xyzの行数
#define XYZ_COL 4		// xyzの列数
#define DATA_ROW 471	// 計算で使用するデータの行数 (390 - 830 nm)
#define DATA_MIN 360	// 使用する周波数の最小値
#define DATA_MAX 830	// 使用する周波数の最大値
#define PI 3.141592		// 円周率

#define BLOCKSIZE 471		// 1ブロック当たりのスレッド数
#define DATANUM 50			// 計算する数
#define CALCNUM 15000		// べき乗する数
#define SIMNUM 1000	    	// シミュレーションする回数
#define LOOPNUM 20			// SIMNUM回のシミュレーション繰り返す回数


#define GAUSS_CNT 10        // 足し合わせるガウシアンの数
#define GAUSS_PER 3         // ガウシアンのパラメータ数

#define MU_MIN  360         // μの最小値
#define MU_MAX  830         // μの最大値
#define TARGET_MU 500       // μの固定値
#define TARGET_SIG 50      // σの固定値

using namespace std;


/* CUDAエラーチェック */
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/* ファイルからデータを読み込む関数 
   d65, CIE LMS, CIE xyz を読み込む*/
int getFileData(double* d65, 
                double* obs_l, double* obs_m, double* obs_s,
                double* obs_x, double* obs_y, double* obs_z) {
    
    /* ファイルポインタ */
    FILE* fp_d65, * fp_obs, * fp_xyz;
    /* EOFを検出する変数 */
    int ret;
    /* カウンター */
    int count = 0;

    /*********************************************************************/
    /* D65の読み込み */
    /* ファイルオープン */
    fp_d65 = fopen("./d65.csv", "r");
    /* 正しく開けているかをチェック */
    if (fp_d65 == NULL) {
        cout << "File open error" << endl;
        return -1;
    }

    for (int i = 0; i < D65_ROW; i++) {
        /* 1時的に波長とデータを格納する変数 */
        double tmp_spt = 0, tmp_data = 0;
        /* 1行ずつ読み込む */
        ret = fscanf(fp_d65, "%lf, %lf", &tmp_spt, &tmp_data);
        /* 終了判定 */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* カウンタ更新 */
        if (tmp_spt >= DATA_MIN) {
            d65[count] = tmp_data;
            count++;
        }
        /* エラーを検出した際の処理 */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_d65);
    count = 0;
    /*********************************************************************/


    /*********************************************************************/
    /* 標準観測者(CIE LMS)の読み込み */
    /* ファイルオープン */
    fp_obs = fopen("./std_obs_10deg.csv", "r");
    /* 正しく開けているかをチェック */
    if (fp_obs == NULL) {
        cout << "File open error" << endl;
        return -1;
    }

    /* ファイル読み込み */
    for (int i = 0; i < (DATA_ROW - OBS_ROW); i++) {
        obs_l[i] = 0;
        obs_m[i] = 0;
        obs_s[i] = 0;
        count++;
    }
    for (int i = 0; i < OBS_ROW; i++) {
        /* 1時的に波長とデータを格納する変数 */
        double tmp_spt = 0, tmp_l = 0, tmp_m = 0, tmp_s = 0;
        /* 1行ずつ読み込む */
        ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &tmp_spt, &tmp_l, &tmp_m, &tmp_s);
        /* 終了判定 */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* カウンタの更新 */
        if (tmp_spt >= DATA_MIN) {
            obs_l[count] = tmp_l;
            obs_m[count] = tmp_m;
            obs_s[count] = tmp_s;
            count++;
        }
        /* エラーを検出した際の処理 */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_obs);
    count = 0;

    /*for (int i = 0; i < DATA_ROW; i++) {
        printf("%lf %lf %lf\n", obs_l[i], obs_m[i], obs_s[i]);
    }*/
    /*********************************************************************/
    

    /*********************************************************************/
    /* xyzの読み込み */
    /* ファイルオープン */
    fp_xyz = fopen("./ciexyz31_v2.csv", "r");
    /* 正しく開けているかをチェック */
    if (fp_xyz == NULL) {
        cout << "File open error" << endl;
        return -1;
    }
    /* ファイル読み込み */
    for (int i = 0; i < XYZ_ROW; i++) {
        /* 1時的に波長とデータを格納する変数 */
        double tmp_spt = 0, tmp_x = 0, tmp_y = 0, tmp_z = 0;
        /* 1行ずつ読み込む */
        ret = fscanf(fp_xyz, "%lf, %lf, %lf, %lf", &tmp_spt, &tmp_x, &tmp_y, &tmp_z);
        /* 終了判定 */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* カウンタの更新 */
        if (tmp_spt >= DATA_MIN) {
            obs_x[count] = tmp_x;
            obs_y[count] = tmp_y;
            obs_z[count] = tmp_z;
            count++;
        }
        /* エラーを検出した際の処理 */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_xyz);

    return 0;
    /*********************************************************************/
}

/* 総和計算の時に使用する変数を計算 */
int getRemain(void) {
    /* 余り */
    int remain = 0;

    /* 余り計算 */
    for (int i = 1; i < BLOCKSIZE; i *= 2) {
        remain = BLOCKSIZE - i;
    }

    /* 余り出力 */
    return remain;
}

/* ガウシアン生成 */
void calcGauss(double* gauss_data) {
    /* 正規分布の確率密度関数を設定する */
    random_device seed_gen;
    default_random_engine generator(seed_gen());
    
    /* TARGET_MU と TARGET_SIG  */
    /*int T_MU[12] = { 480, 480, 480, 530, 530, 530, 580, 580, 580,630,630,630 };
    int T_SIG[12] = { 50,100,200,50,100,200,50,100,200,50,100,200 };*/
    /*int T_MU[4] = { 480, 530, 580, 630 };
    int T_SIG[4] = { 200,200,200,200 };*/
    int T_MU[4] = { 480,480,580,580};
    int T_SIG[4] = { 25,50,25,50 };

    /* カウンタ */
    int count = 0;

    /* 乱数のシード生成 */
    srand((unsigned int)time(NULL));
    for (int j = 0; j < (LOOPNUM / 10); j++) {
        /* 正規分布の確率密度関数を設定する */
        normal_distribution<double> distribution1(T_MU[j], T_SIG[j]);
        normal_distribution<double> distribution2(T_MU[j+2], T_SIG[j+2]);
        for (int i = 0; i < (SIMNUM * 10 * GAUSS_CNT * GAUSS_PER); i += 3) {
            double mu = 0;      // μを初期化
            ///* 10回に1回はμを固定する */
            //if (i % 10 == 0) {
            //    mu = (double)TARGET_MU;
            //}
            //else {
            //    /* μ */
            //    mu = MU_MIN + ((double)rand() / RAND_MAX * (double)(MU_MAX - MU_MIN));
            //}

            double mu_dec = (double)rand() / RAND_MAX;
            if (mu_dec > 0.5) {
                /* muを正規分布の確率密度関数によって生成する */
                mu = (double)distribution1(generator);
            }
            else {
                /* muを正規分布の確率密度関数によって生成する */
                mu = (double)distribution2(generator);
            }
            /* σ */
            double sigma = 5 + (95 * (double)rand() / RAND_MAX);
            /* 振幅の最大値 */
            double g_amp = (double)rand() / RAND_MAX;
            /* gauss_dataに格納 */
            gauss_data[count] = mu;
            gauss_data[count + 1] = sigma;
            gauss_data[count + 2] = g_amp;
            count += 3;
        }
    }
}

/* xyY計算カーネル */
template<int BLOCK_SIZE> __global__ void colorSim(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt,double d_min) {
    /* CUDAアクセス用変数 */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

    /* 結果を格納するシェアードメモリ */
    __shared__ double calc_data[BLOCK_SIZE][3];

    /* 白色点を格納するシェアードメモリ */
    __shared__ double w_point[3];

    /* ガウシアンを足し合わせたものを格納する変数 */
    __shared__ double g_sum[BLOCK_SIZE];
    g_sum[ix] = 0;
    /* ブロック内のスレッド同期 */
    __syncthreads();

    __shared__ double g_tmp[BLOCK_SIZE];
    /* ガウシアンの最大値を保存する変数 */
    __shared__ double tmp_max;
    tmp_max = 0;

    /* ガウシアンのμ */
    double mu;
    /* ガウシアンのσ */
    double sigma;
    /* 振幅の倍率 */
    double g_amp;

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* ガウシアンの足し合わせを行う */
    for (int i = 0; i < g_cnt; i++) {
        /* ガウシアンのμ */
        mu = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i)];
        /* ガウシアンのσ */
        sigma = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 1];
        /* 振幅の倍率 */
        g_amp = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 2];
        /* ガウシアンを一時的に格納 */
        g_tmp[ix] = (1 / (sqrt(2 * pi) * sigma)) * exp((-1) * (((double)ix + d_min) - mu) * (((double)ix + d_min) - mu) / (2 * sigma * sigma));

        /* ブロック内のスレッド同期 */
        __syncthreads();

        /* 最大値を探す */
        if (ix == 0) {
            /* 最大値初期化 */
            tmp_max = 0;
            /* 全データを探索する */
            for (int j = 0; j < BLOCK_SIZE; j++) {
                /* 最大値更新 */
                if (tmp_max < g_tmp[j]) {
                    tmp_max = g_tmp[j];
                }
            }
        }

        /* ブロック内のスレッド同期 */
        __syncthreads();

        /* 最大値が小さすぎた場合(0.01以下）g_ampを0にする */
        if (tmp_max <= 0.01) {
            g_amp = 0;
        }

        /* ブロック内のスレッド同期 */
        __syncthreads();

        /* ガウシアンを足し合わせる */
        g_sum[ix] = g_sum[ix] + (g_tmp[ix] / tmp_max * g_amp);

        /* ブロック内のスレッド同期 */
        __syncthreads();
    }

    /* 足し合わせたガウシアンを正規化する */
    if (ix == 0) {
        /* 最大値を初期化 */
        tmp_max = 0;
        /* 最大値を探す */
        for (int i = 0; i < BLOCK_SIZE; i++) {
            /* 最大値更新 */
            if (tmp_max < g_sum[i]) {
                tmp_max = g_sum[i];
            }
        }
    }

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* 正規化する(0.99で正規化) */
    g_sum[ix] = g_sum[ix] / tmp_max * 0.99;

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* 積分計算をする */
    for (int i = 0; i < CALCNUM; i++) {
        /* シェアードメモリにデータ格納 */
        calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));

        /* ブロック同期 */
        __syncthreads();

        /* ブロックごとにリダクション処理(総和計算) */
        /* 余りが0でない場合 */
        if (remain != 0) {
            /* 余った要素のシェアードメモリを加算する */
            if (ix < remain) {
                calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
                calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
                calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
            }
        }

        /* 総和計算する */
        if (BLOCK_SIZE >= 512) {
            if (ix < 256) {
                calc_data[ix][0] += calc_data[ix + 256][0];
                calc_data[ix][1] += calc_data[ix + 256][1];
                calc_data[ix][2] += calc_data[ix + 256][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 256) {
            if (ix < 128) {
                calc_data[ix][0] += calc_data[ix + 128][0];
                calc_data[ix][1] += calc_data[ix + 128][1];
                calc_data[ix][2] += calc_data[ix + 128][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 128) {
            if (ix < 64) {
                calc_data[ix][0] += calc_data[ix + 64][0];
                calc_data[ix][1] += calc_data[ix + 64][1];
                calc_data[ix][2] += calc_data[ix + 64][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 64) {
            if (ix < 32) {
                calc_data[ix][0] += calc_data[ix + 32][0];
                calc_data[ix][1] += calc_data[ix + 32][1];
                calc_data[ix][2] += calc_data[ix + 32][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 32) {
            if (ix < 16) {
                calc_data[ix][0] += calc_data[ix + 16][0];
                calc_data[ix][1] += calc_data[ix + 16][1];
                calc_data[ix][2] += calc_data[ix + 16][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 16) {
            if (ix < 8) {
                calc_data[ix][0] += calc_data[ix + 8][0];
                calc_data[ix][1] += calc_data[ix + 8][1];
                calc_data[ix][2] += calc_data[ix + 8][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 8) {
            if (ix < 4) {
                calc_data[ix][0] += calc_data[ix + 4][0];
                calc_data[ix][1] += calc_data[ix + 4][1];
                calc_data[ix][2] += calc_data[ix + 4][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 4) {
            if (ix < 2) {
                calc_data[ix][0] += calc_data[ix + 2][0];
                calc_data[ix][1] += calc_data[ix + 2][1];
                calc_data[ix][2] += calc_data[ix + 2][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 2) {
            if (ix < 1) {
                calc_data[ix][0] += calc_data[ix + 1][0];
                calc_data[ix][1] += calc_data[ix + 1][1];
                calc_data[ix][2] += calc_data[ix + 1][2];
            } __syncthreads();
        }

        /* 値出力 */
        if (ix == 0) {
            /* 0割を防止 */
            if (calc_data[0][0] + calc_data[0][1] + calc_data[0][2] > 0.0000000001) {
                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i;
                result[aPos] = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);

                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
                result[aPos] = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);

                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
                result[aPos] = calc_data[0][1];
            }
            else {
                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i;
                result[aPos] = 0.0;

                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
                result[aPos] = 0.0;

                /* aPos更新 */
                aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
                result[aPos] = 0.0;
            }
            //printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの番号をふる */
        /* メッシュ用の変数 */
        double x, y;
        /* フラグxy */
        int f_x, f_y;
        /* メッシュ判定用の変数 */
        double m_x, m_y;

        /* f_x,f_yの初期化 */
        f_x = 0;
        f_y = 0;

        /* aPos初期化 */
        aPos = 0;

        /* x方向 */
        if (ix <= 36) {
            /* xyの計算 */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
            /* メッシュの判定 */
            m_x = (double)ix * 0.02;
            if (m_x <= x && (m_x + 0.02) > x) {
                f_x = 1;
            }
        }
        /* y方向 */
        if (ix >= 64 && ix <= 106) {
            /* xyの計算 */
            y = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
            /* メッシュの判定 */
            m_y = (double)(ix - 64) * 0.02;
            if (m_y <= y && (m_y + 0.02) > y) {
                f_y = 1;
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの位置計算 */
        if (ix <= 36) {
            if (f_x == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = ix;
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの位置計算 */
        if (ix >= 64 && ix <= 106) {
            if (f_y == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] += (ix - 64) * 37;
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* シミュレーションが打ち切られているかを判定 */
        if (ix == 128) {
            /* 繰り返しの最初じゃない場合 */
            if (i > 0) {
                /* XYZの値が白色点の0.5%未満だったとき */
                if ((w_point[0] * 0.005) > calc_data[0][0] &&
                    (w_point[1] * 0.005) > calc_data[0][1] &&
                    (w_point[2] * 0.005) > calc_data[0][2]) {
                    aPos = blockIdx.x * CALCNUM + i;
                    d_mesh[aPos] = -1;
                }
            }
            /* 繰り返しの最初の場合 */
            else {
                w_point[0] = calc_data[0][0];
                w_point[1] = calc_data[0][1];
                w_point[2] = calc_data[0][2];
            }
        }
        /* ブロック同期 */
        __syncthreads();
    }
}
/* LMS計算カーネル */
template<int BLOCK_SIZE> __global__ void colorSimLMS(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt, double d_min) {
    /* CUDAアクセス用変数 */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

    /* 結果を格納するシェアードメモリ */
    __shared__ double calc_data[BLOCK_SIZE][3];

    /* 白色点を格納するシェアードメモリ */
    __shared__ double w_point[3];

    /* ガウシアンを足し合わせたものを格納する変数 */
    __shared__ double g_sum[BLOCK_SIZE];
    g_sum[ix] = 0;
    /* ブロック内のスレッド同期 */
    __syncthreads();

    __shared__ double g_tmp[BLOCK_SIZE];
    /* ガウシアンの最大値を保存する変数 */
    __shared__ double tmp_max;
    tmp_max = 0;

    /* ガウシアンのμ */
    double mu;
    /* ガウシアンのσ */
    double sigma;
    /* 振幅の倍率 */
    double g_amp;

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* ガウシアンの足し合わせを行う */
    for (int i = 0; i < g_cnt; i++) {
        /* ガウシアンのμ */
        mu = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i)];
        /* ガウシアンのσ */
        sigma = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 1];
        /* 振幅の倍率 */
        g_amp = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 2];
        /* ガウシアンを一時的に格納 */
        g_tmp[ix] = (1 / (sqrt(2 * pi) * sigma)) * exp((-1) * (((double)ix + d_min) - mu) * (((double)ix + d_min) - mu) / (2 * sigma * sigma));

        /* ブロック内のスレッド同期 */
        __syncthreads();

        /* 最大値を探す */
        if (ix == 0) {
            /* 最大値初期化 */
            tmp_max = 0;
            /* 全データを探索する */
            for (int j = 0; j < BLOCK_SIZE; j++) {
                /* 最大値更新 */
                if (tmp_max < g_tmp[j]) {
                    tmp_max = g_tmp[j];
                }
            }
        }

        /* ブロック内のスレッド同期 */
        __syncthreads();

        /* ガウシアンを足し合わせる */
        g_sum[ix] = g_sum[ix] + (g_tmp[ix] / tmp_max * g_amp);

        /* ブロック内のスレッド同期 */
        __syncthreads();
    }

    /* 足し合わせたガウシアンを正規化する */
    if (ix == 0) {
        /* 最大値を初期化 */
        tmp_max = 0;
        /* 最大値を探す */
        for (int i = 0; i < BLOCK_SIZE; i++) {
            /* 最大値更新 */
            if (tmp_max < g_sum[i]) {
                tmp_max = g_sum[i];
            }
        }
    }

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* 正規化する(0.99で正規化) */
    g_sum[ix] = g_sum[ix] / tmp_max * 0.99;

    /* ブロック内のスレッド同期 */
    __syncthreads();

    /* 積分計算をする */
    for (int i = 0; i < CALCNUM; i++) {
        /* シェアードメモリにデータ格納 */
        calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));

        /* ブロック同期 */
        __syncthreads();

        /* ブロックごとにリダクション処理(総和計算) */
        /* 余りが0でない場合 */
        if (remain != 0) {
            /* 余った要素のシェアードメモリを加算する */
            if (ix < remain) {
                calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
                calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
                calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
            }
        }

        /* 総和計算する */
        if (BLOCK_SIZE >= 512) {
            if (ix < 256) {
                calc_data[ix][0] += calc_data[ix + 256][0];
                calc_data[ix][1] += calc_data[ix + 256][1];
                calc_data[ix][2] += calc_data[ix + 256][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 256) {
            if (ix < 128) {
                calc_data[ix][0] += calc_data[ix + 128][0];
                calc_data[ix][1] += calc_data[ix + 128][1];
                calc_data[ix][2] += calc_data[ix + 128][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 128) {
            if (ix < 64) {
                calc_data[ix][0] += calc_data[ix + 64][0];
                calc_data[ix][1] += calc_data[ix + 64][1];
                calc_data[ix][2] += calc_data[ix + 64][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 64) {
            if (ix < 32) {
                calc_data[ix][0] += calc_data[ix + 32][0];
                calc_data[ix][1] += calc_data[ix + 32][1];
                calc_data[ix][2] += calc_data[ix + 32][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 32) {
            if (ix < 16) {
                calc_data[ix][0] += calc_data[ix + 16][0];
                calc_data[ix][1] += calc_data[ix + 16][1];
                calc_data[ix][2] += calc_data[ix + 16][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 16) {
            if (ix < 8) {
                calc_data[ix][0] += calc_data[ix + 8][0];
                calc_data[ix][1] += calc_data[ix + 8][1];
                calc_data[ix][2] += calc_data[ix + 8][2];
            }__syncthreads();
        }
        if (BLOCK_SIZE >= 8) {
            if (ix < 4) {
                calc_data[ix][0] += calc_data[ix + 4][0];
                calc_data[ix][1] += calc_data[ix + 4][1];
                calc_data[ix][2] += calc_data[ix + 4][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 4) {
            if (ix < 2) {
                calc_data[ix][0] += calc_data[ix + 2][0];
                calc_data[ix][1] += calc_data[ix + 2][1];
                calc_data[ix][2] += calc_data[ix + 2][2];
            } __syncthreads();
        }
        if (BLOCK_SIZE >= 2) {
            if (ix < 1) {
                calc_data[ix][0] += calc_data[ix + 1][0];
                calc_data[ix][1] += calc_data[ix + 1][1];
                calc_data[ix][2] += calc_data[ix + 1][2];
            } __syncthreads();
        }

        /* 値出力 */
        if (ix == 0) {
            /* aPos更新 */
            aPos = blockIdx.x * 3 * CALCNUM + i;
            result[aPos] = calc_data[0][0];

            /* aPos更新 */
            aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
            result[aPos] = calc_data[0][1];

            /* aPos更新 */
            aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
            result[aPos] = calc_data[0][2];

            //printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの番号をふる */
        /* メッシュ用の変数 */
        double x, y;
        /* フラグxy */
        int f_x, f_y,f_exp;
        /* メッシュ判定用の変数 */
        double m_x, m_y;

        /* f_x,f_yの初期化 */
        f_x = 0;
        f_y = 0;
        f_exp = 0;  // 範囲を超えているものを判定

        /* aPos初期化 */
        aPos = 0;

        /* x方向 */
        if (ix < 50) {
            //printf("kita\n");
            /* xyの計算 */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1]);
            /* メッシュの判定 */
            m_x = (double)ix * 0.02;
            if (m_x <= x && (m_x + 0.02) > x) {
                f_x = 1;
            }
        }
        /* y方向 */
        if (ix >= 64 && ix < 114) {
            /* xyの計算 */
            y = calc_data[0][2] / (calc_data[0][0] + calc_data[0][1]);
            /* メッシュの判定 */
            m_y = (double)(ix - 64) * 0.02;
            if (m_y <= y && (m_y + 0.02) > y) {
                f_y = 1;
            }
        }

        /* 範囲を超えているものを判定 */
        if (ix == 128) {
            /* xyの計算 */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1]);
            y = calc_data[0][2] / (calc_data[0][0] + calc_data[0][1]);
            /* メッシュの判定 */
            if (x > 1 || y > 1) {
                f_exp = 1;
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの位置計算 */
        if (ix < 50) {
            if (f_x == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = ix;
                //printf("%lf %lf %d\n", x, y, d_mesh[aPos]);
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュの位置計算 */
        if (ix >= 64 && ix < 114) {
            if (f_y == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] += (ix - 64) * 50;
                /*printf("%lf %lf %d\n", x, y, d_mesh[aPos]);*/
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* メッシュ範囲外計算 */
        if (ix == 128) {
            if (f_exp == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = -2;
            }
        }

        /* ブロック同期 */
        __syncthreads();

        /* シミュレーションが打ち切られているかを判定 */
        if (ix == 128) {
            /* 繰り返しの最初じゃない場合 */
            if (i > 0) {
                /* XYZの値が白色点の0.5%未満だったとき */
                if ((w_point[0] * 0.005) > calc_data[0][0] &&
                    (w_point[1] * 0.005) > calc_data[0][1] &&
                    (w_point[2] * 0.005) > calc_data[0][2]) {
                    aPos = blockIdx.x * CALCNUM + i;
                    d_mesh[aPos] = -1;
                }
            }
            /* 繰り返しの最初の場合 */
            else {
                w_point[0] = calc_data[0][0];
                w_point[1] = calc_data[0][1];
                w_point[2] = calc_data[0][2];
            }
        }
        /* ブロック同期 */
        __syncthreads();
    }
}

int main(void) {
    /* 出力ディレクトリ */
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v1/";
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v2/";
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_15000_10_v1/";
    //string directory = "G:/isomura_data/sim_result/sim_1000_15000_10_v2/";
    string directory = "G:/isomura_data/sim_result/sim_1000_15000_mu_480_580/";

    /* 出力したファイルの情報を記録するファイル */
    string f_info = "sim_file_info.txt";
    string spt_info = "spectral.txt";
    f_info = directory + f_info;
    spt_info = directory + spt_info;
    ofstream o_f_info(f_info);
    ofstream o_spt_info(spt_info);

    /* スペクトル情報を出力 */
    o_spt_info << "gaussian spectral range" << endl;
    o_spt_info << MU_MIN << "-" << MU_MAX << endl;

    /* データを入れる１次元配列 */
    double* d65, * obs_x, * obs_y, * obs_z, * obs_l, * obs_m, * obs_s, * gauss_data, * result, * fin_result, * lms_result, * lms_fin;
    int* mesh_result, * mesh_f_result, * lms_mesh;
    /* 配列のメモリ確保 */
    d65 = new double[DATA_ROW];
    obs_l = new double[DATA_ROW];
    obs_m = new double[DATA_ROW];
    obs_s = new double[DATA_ROW];
    obs_x = new double[DATA_ROW];
    obs_y = new double[DATA_ROW];
    obs_z = new double[DATA_ROW];
    gauss_data = new double[SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER];
    result = new double[3 * DATANUM * CALCNUM];
    fin_result = new double[3 * SIMNUM * CALCNUM];
    mesh_result = new int[DATANUM * CALCNUM];
    mesh_f_result = new int[SIMNUM * CALCNUM];
    lms_result = new double[3 * DATANUM * CALCNUM];
    lms_fin = new double[3 * SIMNUM * CALCNUM];
    lms_mesh = new int[SIMNUM * CALCNUM];

    /* ファイル書き込み時に使用する変数 */
    double x = 0, y = 0, z = 0;

    /* ファイル読み込み関数実行 */
    int f_result = getFileData(d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z);

    /* ガウシアンの要素を生成 */
    calcGauss(gauss_data);

    /* ガウシアンのデータを出力 */
    string g_info = "gaussian_data.csv";
    g_info = directory + g_info;
    ofstream o_g_info(g_info);
    for (int i = 0; (i < SIMNUM * LOOPNUM * GAUSS_CNT); i++) {
        o_g_info << gauss_data[3 * i] << "," << gauss_data[(3 * i) + 1] << "," << gauss_data[(3 * i) + 2] << endl;
    }

    /* 余り計算 */
    int remain = getRemain();

    /* CUDA用の変数 */
    double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result;
    int* d_mesh;

    /* 結果コピーのときのカウンタ */
    int mem_cnt = 0;    // 積分結果コピー用
    int mesh_cnt = 0;   // メッシュデータコピー用

    /* GPUメモリ確保 */
    cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double));
    cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));
    cudaMalloc((void**)&d_mesh, DATANUM * CALCNUM * sizeof(int));

    /* CUDAへのメモリコピー */
    cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_x, obs_x, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_y, obs_y, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_z, obs_z, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gauss_data, gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double), cudaMemcpyHostToDevice);

    /* 10回ループ */

    int cnt = 0;

    for (int i = 0; i < LOOPNUM; i++) {
        /* 出力ファイル名 */
        string fname1 = "sim_result_xyY_1000_";
        string fname3 = "sim_result_lms_1000_";
        string fend = ".csv";
        //int fnum = (cnt * 30) + (i + 21) - (cnt * 10);
        /*fname1 = directory + fname1 + to_string((cnt * 30) + i + 21 - ((cnt - 1) * 10)) + fend;
        fname3 = directory + fname3 + to_string((cnt * 30) + i + 21 - ((cnt - 1) * 10)) + fend;*/
        /*fname1 = directory + fname1 + to_string(fnum) + fend;
        fname3 = directory + fname3 + to_string(fnum) + fend;*/
        fname1 = directory + fname1 + to_string(i+1) + fend;
        fname3 = directory + fname3 + to_string(i+1) + fend;
        if ((i + 1) % 10 == 0) {
            cnt++;
        }
        /*fname1 = directory + fname1 + to_string(i + 1) + fend;
        fname3 = directory + fname3 + to_string(i + 1) + fend;*/

        /* 出力したファイルの情報を記録するファイルにファイル名を出力 */
        o_f_info << fname1 << endl;
        o_f_info << fname3 << endl;

        /* ファイルオープン */
        FILE* fp_xyz;     // XYZを出力するファイル
        fp_xyz = fopen(fname1.c_str(), "w");      // ファイルオープン

        /* 1000回(DATANUMずつ増加する)ループ */
        for (int j = 0; j < SIMNUM; j += DATANUM) {
            int sim_pos = (i * SIMNUM) + j;
            /* 積分計算 */
            colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > (sim_pos, d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);
            cudaDeviceSynchronize();
            /* 結果のコピー */
            cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);

            /* 結果のファイル出力 */
            for (int k = 0; k < DATANUM; k++) {
                /* xの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_xyz, "\n");
                /* yの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_xyz, "\n");
                /* Yの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_xyz, "\n");
                /* meshの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%d,", mesh_result[mesh_cnt]);
                    mesh_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_xyz, "\n");
            }
            /* カウンタ初期化 */
            mesh_cnt = 0;
            mem_cnt = 0;
        }
        fclose(fp_xyz);
    }

    /* LMS計算に使わないメモリの開放 */
    cudaFree(d_obs_x);
    cudaFree(d_obs_y);
    cudaFree(d_obs_z);

    /* LMS計算で使用するCUDAメモリの確保 */
    double* d_obs_l, * d_obs_m, * d_obs_s;
    cudaMalloc((void**)&d_obs_l, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_m, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_s, DATA_ROW * sizeof(double));
    cudaMemcpy(d_obs_l, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_m, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_s, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);

    /* カウンタ更新 */
    mem_cnt = 0;    // 積分結果コピー用
    mesh_cnt = 0;   // メッシュデータコピー用

    /* LMS計算 */
    for (int i = 0; i < LOOPNUM; i++) {
        /* ファイル名 */
        string fname3 = "sim_result_lms_1000_";
        string fend = ".csv";
        fname3 = directory + fname3 + to_string(i + 1) + fend;

        /* ファイルオープン */
        FILE* fp_lms;     // XYZを出力するファイル
        fp_lms = fopen(fname3.c_str(), "w");      // ファイルオープン

        for (int j = 0; j < SIMNUM; j += DATANUM) {
            int sim_pos = (i * SIMNUM) + j;
            /* 積分計算 */
            colorSimLMS<DATA_ROW> << <DATANUM, DATA_ROW >> > (sim_pos, d_gauss_data, d_d65, d_obs_l, d_obs_m, d_obs_s, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);
            cudaDeviceSynchronize();
            /* 結果のコピー */
            cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);
            /* 結果のファイル出力 */
            for (int k = 0; k < DATANUM; k++) {
                /* xの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_lms, "\n");
                /* yの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_lms, "\n");
                /* Yの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_lms, "\n");
                /* meshの出力 */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%d,", mesh_result[mesh_cnt]);
                    mesh_cnt++;
                }
                /* 改行部分 */
                fprintf(fp_lms, "\n");
            }
            /* カウンタ初期化 */
            mesh_cnt = 0;
            mem_cnt = 0;
        }
        fclose(fp_lms);
    }

    /* メモリ解放 */
    cudaFree(d_result);
    cudaFree(d_mesh);
    cudaFree(d_d65);
    cudaFree(d_gauss_data);
    cudaFree(d_obs_x);
    cudaFree(d_obs_y);
    cudaFree(d_obs_z);
    cudaFree(d_obs_l);
    cudaFree(d_obs_m);
    cudaFree(d_obs_s);

    /* ホストメモリ解放 */
    delete[] d65;
    delete[] obs_x;
    delete[] obs_y;
    delete[] obs_z;
    delete[] obs_l;
    delete[] obs_m;
    delete[] obs_s;
    delete[] gauss_data;
    delete[] result;
    delete[] fin_result;
    delete[] lms_result;
    delete[] lms_fin;

    return 0;
}