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

#define D65_ROW 531		// D65�̍s��
#define D65_COL 2		// D65�̗�
#define OBS_ROW 441		// �W���ϑ��҂̍s��
#define OBS_COL 4		// �W���ϑ��҂̗�
#define XYZ_ROW 471		// xyz�̍s��
#define XYZ_COL 4		// xyz�̗�
#define DATA_ROW 441	// �v�Z�Ŏg�p����f�[�^�̍s�� (390 - 830 nm)
#define DATA_MIN 390	// �g�p������g���̍ŏ��l
#define DATA_MAX 830	// �g�p������g���̍ő�l
#define PI 3.141592		// �~����

#define BLOCKSIZE 441		// 1�u���b�N������̃X���b�h��
#define DATANUM 50			// �v�Z���鐔
#define CALCNUM 25000		// �ׂ��悷�鐔
#define SIMNUM 1000			// �V�~�����[�V���������
#define LOOPNUM 10			// SIMNUM��̃V�~�����[�V�����J��Ԃ���
#define GAUSS_CNT 10        // �������킹��K�E�V�A���̐�
#define GAUSS_PER 3         // �K�E�V�A���̃p�����[�^��

#define MU_MIN  390         // �ʂ̍ŏ��l
#define MU_MAX  830         // �ʂ̍ő�l

using namespace std;

/* CUDA�G���[�`�F�b�N */
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

/* �t�@�C������f�[�^��ǂݍ��ފ֐� 
   d65, CIE LMS, CIE xyz ��ǂݍ���*/
int getFileData(double* d65, 
                double* obs_l, double* obs_m, double* obs_s,
                double* obs_x, double* obs_y, double* obs_z) {
    
    /* �t�@�C���|�C���^ */
    FILE* fp_d65, * fp_obs, * fp_xyz;
    /* EOF�����o����ϐ� */
    int ret;
    /* �J�E���^�[ */
    int count = 0;

    /*********************************************************************/
    /* D65�̓ǂݍ��� */
    /* �t�@�C���I�[�v�� */
    fp_d65 = fopen("./d65.csv", "r");
    /* �������J���Ă��邩���`�F�b�N */
    if (fp_d65 == NULL) {
        cout << "File open error" << endl;
        return -1;
    }

    for (int i = 0; i < D65_ROW; i++) {
        /* 1���I�ɔg���ƃf�[�^���i�[����ϐ� */
        double tmp_spt = 0, tmp_data = 0;
        /* 1�s���ǂݍ��� */
        ret = fscanf(fp_d65, "%lf, %lf", &tmp_spt, &tmp_data);
        /* �I������ */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* �J�E���^�X�V */
        if (tmp_spt >= DATA_MIN) {
            d65[count] = tmp_data;
            count++;
        }
        /* �G���[�����o�����ۂ̏��� */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_d65);
    count = 0;
    /*********************************************************************/


    /*********************************************************************/
    /* �W���ϑ���(CIE LMS)�̓ǂݍ��� */
    /* �t�@�C���I�[�v�� */
    fp_obs = fopen("./std_obs_10deg.csv", "r");
    /* �������J���Ă��邩���`�F�b�N */
    if (fp_obs == NULL) {
        cout << "File open error" << endl;
        return -1;
    }

    /* �t�@�C���ǂݍ��� */
    for (int i = 0; i < OBS_ROW; i++) {
        /* 1���I�ɔg���ƃf�[�^���i�[����ϐ� */
        double tmp_spt = 0, tmp_l = 0, tmp_m = 0, tmp_s = 0;
        /* 1�s���ǂݍ��� */
        ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &tmp_spt, &tmp_l, &tmp_m, &tmp_s);
        /* �I������ */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* �J�E���^�̍X�V */
        if (tmp_spt >= DATA_MIN) {
            obs_l[count] = tmp_l;
            obs_m[count] = tmp_m;
            obs_s[count] = tmp_s;
            count++;
        }
        /* �G���[�����o�����ۂ̏��� */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_obs);
    count = 0;
    /*********************************************************************/
    

    /*********************************************************************/
    /* xyz�̓ǂݍ��� */
    /* �t�@�C���I�[�v�� */
    fp_xyz = fopen("./ciexyz31.csv", "r");
    /* �������J���Ă��邩���`�F�b�N */
    if (fp_xyz == NULL) {
        cout << "File open error" << endl;
        return -1;
    }
    /* �t�@�C���ǂݍ��� */
    for (int i = 0; i < XYZ_ROW; i++) {
        /* 1���I�ɔg���ƃf�[�^���i�[����ϐ� */
        double tmp_spt = 0, tmp_x = 0, tmp_y = 0, tmp_z = 0;
        /* 1�s���ǂݍ��� */
        ret = fscanf(fp_xyz, "%lf, %lf, %lf, %lf", &tmp_spt, &tmp_x, &tmp_y, &tmp_z);
        /* �I������ */
        if (tmp_spt > DATA_MAX) {
            break;
        }
        /* �J�E���^�̍X�V */
        if (tmp_spt >= DATA_MIN) {
            obs_x[count] = tmp_x;
            obs_y[count] = tmp_y;
            obs_z[count] = tmp_z;
            count++;
        }
        /* �G���[�����o�����ۂ̏��� */
        if (ret == EOF) {
            cout << "error" << endl;
            return -1;
        }
    }
    fclose(fp_xyz);

    return 0;
    /*********************************************************************/
}

/* ���a�v�Z�̎��Ɏg�p����ϐ����v�Z */
int getRemain(void) {
    /* �]�� */
    int remain = 0;

    /* �]��v�Z */
    for (int i = 1; i < BLOCKSIZE; i *= 2) {
        remain = BLOCKSIZE - i;
    }

    /* �]��o�� */
    return remain;
}

/* �K�E�V�A������ */
void calcGauss(double* gauss_data) {
    /* �����̃V�[�h���� */
    srand((unsigned int)time(NULL));
    for (int i = 0; i < (SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER); i += 3) {
        /* �� */
        double mu = MU_MIN + ((double)rand() / RAND_MAX * (double)(MU_MAX - MU_MIN));
        /* �� */
        double sigma = 5 + (95 * (double)rand() / RAND_MAX);
        /* �U���̍ő�l */
        double g_amp = (double)rand() / RAND_MAX;
        /* gauss_data�Ɋi�[ */
        gauss_data[i] = mu;
        gauss_data[i + 1] = sigma;
        gauss_data[i + 2] = g_amp;
    }
}

/* xyz�v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void colorSim(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt,double d_min) {
    /* CUDA�A�N�Z�X�p�ϐ� */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

    /* �K�E�V�A���𑫂����킹�����̂��i�[����ϐ� */
    __shared__ double g_sum[BLOCK_SIZE];
    g_sum[ix] = 0;
    /* �u���b�N���̃X���b�h���� */
    __syncthreads();

    __shared__ double g_tmp[BLOCK_SIZE];
    /* �K�E�V�A���̍ő�l��ۑ�����ϐ� */
    __shared__ double tmp_max;
    tmp_max = 0;

    /* �K�E�V�A���̃� */
    double mu;
    /* �K�E�V�A���̃� */
    double sigma;
    /* �U���̔{�� */
    double g_amp;

    /* �u���b�N���̃X���b�h���� */
    __syncthreads();

    /* �K�E�V�A���̑������킹���s�� */
    for (int i = 0; i < g_cnt; i++) {
        /* �K�E�V�A���̃� */
        mu = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i)];
        /* �K�E�V�A���̃� */
        sigma = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 1];
        /* �U���̔{�� */
        g_amp = g_data[((simNum + blockIdx.x) * 3 * g_cnt) + (3 * i) + 2];
        /* �K�E�V�A�����ꎞ�I�Ɋi�[ */
        g_tmp[ix] = (1 / (sqrt(2 * pi) * sigma)) * exp((-1) * (((double)ix + d_min) - mu) * (((double)ix + d_min) - mu) / (2 * sigma * sigma));

        /* �u���b�N���̃X���b�h���� */
        __syncthreads();

        /* �ő�l��T�� */
        if (ix == 0) {
            /* �ő�l������ */
            tmp_max = 0;
            /* �S�f�[�^��T������ */
            for (int j = 0; j < BLOCK_SIZE; j++) {
                /* �ő�l�X�V */
                if (tmp_max < g_tmp[j]) {
                    tmp_max = g_tmp[j];
                }
            }
        }

        /* �u���b�N���̃X���b�h���� */
        __syncthreads();

        /* �K�E�V�A���𑫂����킹�� */
        g_sum[ix] = g_sum[ix] + (g_tmp[ix] / tmp_max * g_amp);

        /* �u���b�N���̃X���b�h���� */
        __syncthreads();
    }

    /* �������킹���K�E�V�A���𐳋K������ */
    if (ix == 0) {
        /* �ő�l�������� */
        tmp_max = 0;
        /* �ő�l��T�� */
        for (int i = 0; i < BLOCK_SIZE; i++) {
            /* �ő�l�X�V */
            if (tmp_max < g_sum[i]) {
                tmp_max = g_sum[i];
            }
        }
    }

    /* �u���b�N���̃X���b�h���� */
    __syncthreads();

    /* ���K������(0.99�Ő��K��) */
    g_sum[ix] = g_sum[ix] / tmp_max * 0.99;

    /* �u���b�N���̃X���b�h���� */
    __syncthreads();
}


int main(void) {
    /* �f�[�^������P�����z�� */
    double* d65, * obs_x, * obs_y, * obs_z, * obs_l, * obs_m, * obs_s, * gauss_data, * result, * fin_result, * lms_result, * lms_fin;
    int* mesh_result, * mesh_f_result;
    /* �z��̃������m�� */
    d65 = new double[DATA_ROW];
    obs_l = new double[DATA_ROW];
    obs_m = new double[DATA_ROW];
    obs_s = new double[DATA_ROW];
    obs_x = new double[DATA_ROW];
    obs_y = new double[DATA_ROW];
    obs_z = new double[DATA_ROW];
    gauss_data = new double[SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER];

    /* �t�@�C���ǂݍ��݊֐����s */
    int f_result = getFileData(d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z);

    /* �K�E�V�A���̗v�f�𐶐� */
    calcGauss(gauss_data);

    /* �]��v�Z */
    int remain = getRemain();

    /* CUDA�p�̕ϐ� */
    double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result, * d_lms;
    int* d_mesh;

    /* GPU�������m�� */
    cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double));
    cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));
    cudaMalloc((void**)&d_mesh, DATANUM * CALCNUM * sizeof(int));

    /* CUDA�ւ̃������R�s�[ */
    cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gauss_data, gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double), cudaMemcpyHostToDevice);

    colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > (0, d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);

    ///* �o�̓f�B���N�g�� */
    //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v1/";
    //string fname = "test.csv";
    //string o_fname = directory + fname;

    ///* �t�@�C���o�̓X�g���[�� */
    //ofstream o_file(o_fname);

    ///* �t�@�C���o�� */
    //for (int i = 0; i < (SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER); i += 3) {
    //    o_file << gauss_data[i] << "," << gauss_data[i+1] << "," << gauss_data[i+2] << endl;
    //}
    return 0;
}