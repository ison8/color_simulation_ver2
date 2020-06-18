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
#define SIMNUM 1000			// シミュレーションする回数
#define LOOPNUM 10			// SIMNUM回のシミュレーション繰り返す回数
#define GAUSS_CNT 10        // 足し合わせるガウシアンの数
#define GAUSS_PER 3         // ガウシアンのパラメータ数

#define MU_MIN  390         // μの最小値
#define MU_MAX  830         // μの最大値

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
    /*********************************************************************/
    

    /*********************************************************************/
    /* xyzの読み込み */
    /* ファイルオープン */
    fp_xyz = fopen("./ciexyz31.csv", "r");
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
    /* 乱数のシード生成 */
    srand((unsigned int)time(NULL));
    for (int i = 0; i < (SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER); i += 3) {
        /* μ */
        double mu = MU_MIN + ((double)rand() / RAND_MAX * (double)(MU_MAX - MU_MIN));
        /* σ */
        double sigma = 5 + (95 * (double)rand() / RAND_MAX);
        /* 振幅の最大値 */
        double g_amp = (double)rand() / RAND_MAX;
        /* gauss_dataに格納 */
        gauss_data[i] = mu;
        gauss_data[i + 1] = sigma;
        gauss_data[i + 2] = g_amp;
    }
}

/* xyz計算カーネル */
template<int BLOCK_SIZE> __global__ void colorSim(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt,double d_min) {
    /* CUDAアクセス用変数 */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

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
}


int main(void) {
    /* データを入れる１次元配列 */
    double* d65, * obs_x, * obs_y, * obs_z, * obs_l, * obs_m, * obs_s, * gauss_data, * result, * fin_result, * lms_result, * lms_fin;
    int* mesh_result, * mesh_f_result;
    /* 配列のメモリ確保 */
    d65 = new double[DATA_ROW];
    obs_l = new double[DATA_ROW];
    obs_m = new double[DATA_ROW];
    obs_s = new double[DATA_ROW];
    obs_x = new double[DATA_ROW];
    obs_y = new double[DATA_ROW];
    obs_z = new double[DATA_ROW];
    gauss_data = new double[SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER];

    /* ファイル読み込み関数実行 */
    int f_result = getFileData(d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z);

    /* ガウシアンの要素を生成 */
    calcGauss(gauss_data);

    /* 余り計算 */
    int remain = getRemain();

    /* CUDA用の変数 */
    double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result, * d_lms;
    int* d_mesh;

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
    cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gauss_data, gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double), cudaMemcpyHostToDevice);

    colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > (0, d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);

    ///* 出力ディレクトリ */
    //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v1/";
    //string fname = "test.csv";
    //string o_fname = directory + fname;

    ///* ファイル出力ストリーム */
    //ofstream o_file(o_fname);

    ///* ファイル出力 */
    //for (int i = 0; i < (SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER); i += 3) {
    //    o_file << gauss_data[i] << "," << gauss_data[i+1] << "," << gauss_data[i+2] << endl;
    //}
    return 0;
}