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

#define D65_ROW 531		// D65�̍s��
#define D65_COL 2		// D65�̗�
#define OBS_ROW 441		// �W���ϑ��҂̍s��
#define OBS_COL 4		// �W���ϑ��҂̗�
#define XYZ_ROW 471		// xyz�̍s��
#define XYZ_COL 4		// xyz�̗�
#define DATA_ROW 471	// �v�Z�Ŏg�p����f�[�^�̍s�� (390 - 830 nm)
#define DATA_MIN 360	// �g�p������g���̍ŏ��l
#define DATA_MAX 830	// �g�p������g���̍ő�l
#define PI 3.141592		// �~����

#define BLOCKSIZE 471		// 1�u���b�N������̃X���b�h��
#define DATANUM 50			// �v�Z���鐔
#define CALCNUM 15000		// �ׂ��悷�鐔
#define SIMNUM 1000	    	// �V�~�����[�V���������
#define LOOPNUM 20			// SIMNUM��̃V�~�����[�V�����J��Ԃ���


#define GAUSS_CNT 10        // �������킹��K�E�V�A���̐�
#define GAUSS_PER 3         // �K�E�V�A���̃p�����[�^��

#define MU_MIN  360         // �ʂ̍ŏ��l
#define MU_MAX  830         // �ʂ̍ő�l
#define TARGET_MU 500       // �ʂ̌Œ�l
#define TARGET_SIG 50      // �Ђ̌Œ�l

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
    for (int i = 0; i < (DATA_ROW - OBS_ROW); i++) {
        obs_l[i] = 0;
        obs_m[i] = 0;
        obs_s[i] = 0;
        count++;
    }
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

    /*for (int i = 0; i < DATA_ROW; i++) {
        printf("%lf %lf %lf\n", obs_l[i], obs_m[i], obs_s[i]);
    }*/
    /*********************************************************************/
    

    /*********************************************************************/
    /* xyz�̓ǂݍ��� */
    /* �t�@�C���I�[�v�� */
    fp_xyz = fopen("./ciexyz31_v2.csv", "r");
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
    /* ���K���z�̊m�����x�֐���ݒ肷�� */
    random_device seed_gen;
    default_random_engine generator(seed_gen());
    
    /* TARGET_MU �� TARGET_SIG  */
    /*int T_MU[12] = { 480, 480, 480, 530, 530, 530, 580, 580, 580,630,630,630 };
    int T_SIG[12] = { 50,100,200,50,100,200,50,100,200,50,100,200 };*/
    /*int T_MU[4] = { 480, 530, 580, 630 };
    int T_SIG[4] = { 200,200,200,200 };*/
    int T_MU[4] = { 480,480,580,580};
    int T_SIG[4] = { 25,50,25,50 };

    /* �J�E���^ */
    int count = 0;

    /* �����̃V�[�h���� */
    srand((unsigned int)time(NULL));
    for (int j = 0; j < (LOOPNUM / 10); j++) {
        /* ���K���z�̊m�����x�֐���ݒ肷�� */
        normal_distribution<double> distribution1(T_MU[j], T_SIG[j]);
        normal_distribution<double> distribution2(T_MU[j+2], T_SIG[j+2]);
        for (int i = 0; i < (SIMNUM * 10 * GAUSS_CNT * GAUSS_PER); i += 3) {
            double mu = 0;      // �ʂ�������
            ///* 10���1��̓ʂ��Œ肷�� */
            //if (i % 10 == 0) {
            //    mu = (double)TARGET_MU;
            //}
            //else {
            //    /* �� */
            //    mu = MU_MIN + ((double)rand() / RAND_MAX * (double)(MU_MAX - MU_MIN));
            //}

            double mu_dec = (double)rand() / RAND_MAX;
            if (mu_dec > 0.5) {
                /* mu�𐳋K���z�̊m�����x�֐��ɂ���Đ������� */
                mu = (double)distribution1(generator);
            }
            else {
                /* mu�𐳋K���z�̊m�����x�֐��ɂ���Đ������� */
                mu = (double)distribution2(generator);
            }
            /* �� */
            double sigma = 5 + (95 * (double)rand() / RAND_MAX);
            /* �U���̍ő�l */
            double g_amp = (double)rand() / RAND_MAX;
            /* gauss_data�Ɋi�[ */
            gauss_data[count] = mu;
            gauss_data[count + 1] = sigma;
            gauss_data[count + 2] = g_amp;
            count += 3;
        }
    }
}

/* xyY�v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void colorSim(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt,double d_min) {
    /* CUDA�A�N�Z�X�p�ϐ� */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

    /* ���ʂ��i�[����V�F�A�[�h������ */
    __shared__ double calc_data[BLOCK_SIZE][3];

    /* ���F�_���i�[����V�F�A�[�h������ */
    __shared__ double w_point[3];

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

        /* �ő�l�������������ꍇ(0.01�ȉ��jg_amp��0�ɂ��� */
        if (tmp_max <= 0.01) {
            g_amp = 0;
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

    /* �ϕ��v�Z������ */
    for (int i = 0; i < CALCNUM; i++) {
        /* �V�F�A�[�h�������Ƀf�[�^�i�[ */
        calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));

        /* �u���b�N���� */
        __syncthreads();

        /* �u���b�N���ƂɃ��_�N�V��������(���a�v�Z) */
        /* �]�肪0�łȂ��ꍇ */
        if (remain != 0) {
            /* �]�����v�f�̃V�F�A�[�h�����������Z���� */
            if (ix < remain) {
                calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
                calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
                calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
            }
        }

        /* ���a�v�Z���� */
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

        /* �l�o�� */
        if (ix == 0) {
            /* 0����h�~ */
            if (calc_data[0][0] + calc_data[0][1] + calc_data[0][2] > 0.0000000001) {
                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i;
                result[aPos] = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);

                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
                result[aPos] = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);

                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
                result[aPos] = calc_data[0][1];
            }
            else {
                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i;
                result[aPos] = 0.0;

                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
                result[aPos] = 0.0;

                /* aPos�X�V */
                aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
                result[aPos] = 0.0;
            }
            //printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̔ԍ����ӂ� */
        /* ���b�V���p�̕ϐ� */
        double x, y;
        /* �t���Oxy */
        int f_x, f_y;
        /* ���b�V������p�̕ϐ� */
        double m_x, m_y;

        /* f_x,f_y�̏����� */
        f_x = 0;
        f_y = 0;

        /* aPos������ */
        aPos = 0;

        /* x���� */
        if (ix <= 36) {
            /* xy�̌v�Z */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
            /* ���b�V���̔��� */
            m_x = (double)ix * 0.02;
            if (m_x <= x && (m_x + 0.02) > x) {
                f_x = 1;
            }
        }
        /* y���� */
        if (ix >= 64 && ix <= 106) {
            /* xy�̌v�Z */
            y = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
            /* ���b�V���̔��� */
            m_y = (double)(ix - 64) * 0.02;
            if (m_y <= y && (m_y + 0.02) > y) {
                f_y = 1;
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̈ʒu�v�Z */
        if (ix <= 36) {
            if (f_x == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = ix;
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̈ʒu�v�Z */
        if (ix >= 64 && ix <= 106) {
            if (f_y == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] += (ix - 64) * 37;
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* �V�~�����[�V�������ł��؂��Ă��邩�𔻒� */
        if (ix == 128) {
            /* �J��Ԃ��̍ŏ�����Ȃ��ꍇ */
            if (i > 0) {
                /* XYZ�̒l�����F�_��0.5%�����������Ƃ� */
                if ((w_point[0] * 0.005) > calc_data[0][0] &&
                    (w_point[1] * 0.005) > calc_data[0][1] &&
                    (w_point[2] * 0.005) > calc_data[0][2]) {
                    aPos = blockIdx.x * CALCNUM + i;
                    d_mesh[aPos] = -1;
                }
            }
            /* �J��Ԃ��̍ŏ��̏ꍇ */
            else {
                w_point[0] = calc_data[0][0];
                w_point[1] = calc_data[0][1];
                w_point[2] = calc_data[0][2];
            }
        }
        /* �u���b�N���� */
        __syncthreads();
    }
}
/* LMS�v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void colorSimLMS(int simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain, int* d_mesh, int g_cnt, double d_min) {
    /* CUDA�A�N�Z�X�p�ϐ� */
    int ix = threadIdx.x;
    int aPos = 0;
    double pi = 3.141592;

    /* ���ʂ��i�[����V�F�A�[�h������ */
    __shared__ double calc_data[BLOCK_SIZE][3];

    /* ���F�_���i�[����V�F�A�[�h������ */
    __shared__ double w_point[3];

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

    /* �ϕ��v�Z������ */
    for (int i = 0; i < CALCNUM; i++) {
        /* �V�F�A�[�h�������Ƀf�[�^�i�[ */
        calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));
        calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(g_sum[ix], (pow(0.001 * (double)i, 2)));

        /* �u���b�N���� */
        __syncthreads();

        /* �u���b�N���ƂɃ��_�N�V��������(���a�v�Z) */
        /* �]�肪0�łȂ��ꍇ */
        if (remain != 0) {
            /* �]�����v�f�̃V�F�A�[�h�����������Z���� */
            if (ix < remain) {
                calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
                calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
                calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
            }
        }

        /* ���a�v�Z���� */
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

        /* �l�o�� */
        if (ix == 0) {
            /* aPos�X�V */
            aPos = blockIdx.x * 3 * CALCNUM + i;
            result[aPos] = calc_data[0][0];

            /* aPos�X�V */
            aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
            result[aPos] = calc_data[0][1];

            /* aPos�X�V */
            aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
            result[aPos] = calc_data[0][2];

            //printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̔ԍ����ӂ� */
        /* ���b�V���p�̕ϐ� */
        double x, y;
        /* �t���Oxy */
        int f_x, f_y,f_exp;
        /* ���b�V������p�̕ϐ� */
        double m_x, m_y;

        /* f_x,f_y�̏����� */
        f_x = 0;
        f_y = 0;
        f_exp = 0;  // �͈͂𒴂��Ă�����̂𔻒�

        /* aPos������ */
        aPos = 0;

        /* x���� */
        if (ix < 50) {
            //printf("kita\n");
            /* xy�̌v�Z */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1]);
            /* ���b�V���̔��� */
            m_x = (double)ix * 0.02;
            if (m_x <= x && (m_x + 0.02) > x) {
                f_x = 1;
            }
        }
        /* y���� */
        if (ix >= 64 && ix < 114) {
            /* xy�̌v�Z */
            y = calc_data[0][2] / (calc_data[0][0] + calc_data[0][1]);
            /* ���b�V���̔��� */
            m_y = (double)(ix - 64) * 0.02;
            if (m_y <= y && (m_y + 0.02) > y) {
                f_y = 1;
            }
        }

        /* �͈͂𒴂��Ă�����̂𔻒� */
        if (ix == 128) {
            /* xy�̌v�Z */
            x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1]);
            y = calc_data[0][2] / (calc_data[0][0] + calc_data[0][1]);
            /* ���b�V���̔��� */
            if (x > 1 || y > 1) {
                f_exp = 1;
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̈ʒu�v�Z */
        if (ix < 50) {
            if (f_x == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = ix;
                //printf("%lf %lf %d\n", x, y, d_mesh[aPos]);
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���̈ʒu�v�Z */
        if (ix >= 64 && ix < 114) {
            if (f_y == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] += (ix - 64) * 50;
                /*printf("%lf %lf %d\n", x, y, d_mesh[aPos]);*/
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* ���b�V���͈͊O�v�Z */
        if (ix == 128) {
            if (f_exp == 1) {
                aPos = blockIdx.x * CALCNUM + i;
                d_mesh[aPos] = -2;
            }
        }

        /* �u���b�N���� */
        __syncthreads();

        /* �V�~�����[�V�������ł��؂��Ă��邩�𔻒� */
        if (ix == 128) {
            /* �J��Ԃ��̍ŏ�����Ȃ��ꍇ */
            if (i > 0) {
                /* XYZ�̒l�����F�_��0.5%�����������Ƃ� */
                if ((w_point[0] * 0.005) > calc_data[0][0] &&
                    (w_point[1] * 0.005) > calc_data[0][1] &&
                    (w_point[2] * 0.005) > calc_data[0][2]) {
                    aPos = blockIdx.x * CALCNUM + i;
                    d_mesh[aPos] = -1;
                }
            }
            /* �J��Ԃ��̍ŏ��̏ꍇ */
            else {
                w_point[0] = calc_data[0][0];
                w_point[1] = calc_data[0][1];
                w_point[2] = calc_data[0][2];
            }
        }
        /* �u���b�N���� */
        __syncthreads();
    }
}

int main(void) {
    /* �o�̓f�B���N�g�� */
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v1/";
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v2/";
   //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_15000_10_v1/";
    //string directory = "G:/isomura_data/sim_result/sim_1000_15000_10_v2/";
    string directory = "G:/isomura_data/sim_result/sim_1000_15000_mu_480_580/";

    /* �o�͂����t�@�C���̏����L�^����t�@�C�� */
    string f_info = "sim_file_info.txt";
    string spt_info = "spectral.txt";
    f_info = directory + f_info;
    spt_info = directory + spt_info;
    ofstream o_f_info(f_info);
    ofstream o_spt_info(spt_info);

    /* �X�y�N�g�������o�� */
    o_spt_info << "gaussian spectral range" << endl;
    o_spt_info << MU_MIN << "-" << MU_MAX << endl;

    /* �f�[�^������P�����z�� */
    double* d65, * obs_x, * obs_y, * obs_z, * obs_l, * obs_m, * obs_s, * gauss_data, * result, * fin_result, * lms_result, * lms_fin;
    int* mesh_result, * mesh_f_result, * lms_mesh;
    /* �z��̃������m�� */
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

    /* �t�@�C���������ݎ��Ɏg�p����ϐ� */
    double x = 0, y = 0, z = 0;

    /* �t�@�C���ǂݍ��݊֐����s */
    int f_result = getFileData(d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z);

    /* �K�E�V�A���̗v�f�𐶐� */
    calcGauss(gauss_data);

    /* �K�E�V�A���̃f�[�^���o�� */
    string g_info = "gaussian_data.csv";
    g_info = directory + g_info;
    ofstream o_g_info(g_info);
    for (int i = 0; (i < SIMNUM * LOOPNUM * GAUSS_CNT); i++) {
        o_g_info << gauss_data[3 * i] << "," << gauss_data[(3 * i) + 1] << "," << gauss_data[(3 * i) + 2] << endl;
    }

    /* �]��v�Z */
    int remain = getRemain();

    /* CUDA�p�̕ϐ� */
    double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result;
    int* d_mesh;

    /* ���ʃR�s�[�̂Ƃ��̃J�E���^ */
    int mem_cnt = 0;    // �ϕ����ʃR�s�[�p
    int mesh_cnt = 0;   // ���b�V���f�[�^�R�s�[�p

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
    cudaMemcpy(d_obs_x, obs_x, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_y, obs_y, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_z, obs_z, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gauss_data, gauss_data, SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER * sizeof(double), cudaMemcpyHostToDevice);

    /* 10�񃋁[�v */

    int cnt = 0;

    for (int i = 0; i < LOOPNUM; i++) {
        /* �o�̓t�@�C���� */
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

        /* �o�͂����t�@�C���̏����L�^����t�@�C���Ƀt�@�C�������o�� */
        o_f_info << fname1 << endl;
        o_f_info << fname3 << endl;

        /* �t�@�C���I�[�v�� */
        FILE* fp_xyz;     // XYZ���o�͂���t�@�C��
        fp_xyz = fopen(fname1.c_str(), "w");      // �t�@�C���I�[�v��

        /* 1000��(DATANUM����������)���[�v */
        for (int j = 0; j < SIMNUM; j += DATANUM) {
            int sim_pos = (i * SIMNUM) + j;
            /* �ϕ��v�Z */
            colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > (sim_pos, d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);
            cudaDeviceSynchronize();
            /* ���ʂ̃R�s�[ */
            cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);

            /* ���ʂ̃t�@�C���o�� */
            for (int k = 0; k < DATANUM; k++) {
                /* x�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_xyz, "\n");
                /* y�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_xyz, "\n");
                /* Y�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_xyz, "\n");
                /* mesh�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_xyz, "%d,", mesh_result[mesh_cnt]);
                    mesh_cnt++;
                }
                /* ���s���� */
                fprintf(fp_xyz, "\n");
            }
            /* �J�E���^������ */
            mesh_cnt = 0;
            mem_cnt = 0;
        }
        fclose(fp_xyz);
    }

    /* LMS�v�Z�Ɏg��Ȃ��������̊J�� */
    cudaFree(d_obs_x);
    cudaFree(d_obs_y);
    cudaFree(d_obs_z);

    /* LMS�v�Z�Ŏg�p����CUDA�������̊m�� */
    double* d_obs_l, * d_obs_m, * d_obs_s;
    cudaMalloc((void**)&d_obs_l, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_m, DATA_ROW * sizeof(double));
    cudaMalloc((void**)&d_obs_s, DATA_ROW * sizeof(double));
    cudaMemcpy(d_obs_l, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_m, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs_s, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);

    /* �J�E���^�X�V */
    mem_cnt = 0;    // �ϕ����ʃR�s�[�p
    mesh_cnt = 0;   // ���b�V���f�[�^�R�s�[�p

    /* LMS�v�Z */
    for (int i = 0; i < LOOPNUM; i++) {
        /* �t�@�C���� */
        string fname3 = "sim_result_lms_1000_";
        string fend = ".csv";
        fname3 = directory + fname3 + to_string(i + 1) + fend;

        /* �t�@�C���I�[�v�� */
        FILE* fp_lms;     // XYZ���o�͂���t�@�C��
        fp_lms = fopen(fname3.c_str(), "w");      // �t�@�C���I�[�v��

        for (int j = 0; j < SIMNUM; j += DATANUM) {
            int sim_pos = (i * SIMNUM) + j;
            /* �ϕ��v�Z */
            colorSimLMS<DATA_ROW> << <DATANUM, DATA_ROW >> > (sim_pos, d_gauss_data, d_d65, d_obs_l, d_obs_m, d_obs_s, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);
            cudaDeviceSynchronize();
            /* ���ʂ̃R�s�[ */
            cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);
            /* ���ʂ̃t�@�C���o�� */
            for (int k = 0; k < DATANUM; k++) {
                /* x�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_lms, "\n");
                /* y�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_lms, "\n");
                /* Y�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%.3lf,", result[mem_cnt]);
                    mem_cnt++;
                }
                /* ���s���� */
                fprintf(fp_lms, "\n");
                /* mesh�̏o�� */
                for (int l = 0; l < CALCNUM; l++) {
                    fprintf(fp_lms, "%d,", mesh_result[mesh_cnt]);
                    mesh_cnt++;
                }
                /* ���s���� */
                fprintf(fp_lms, "\n");
            }
            /* �J�E���^������ */
            mesh_cnt = 0;
            mem_cnt = 0;
        }
        fclose(fp_lms);
    }

    /* ��������� */
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

    /* �z�X�g��������� */
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