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
#define DATA_ROW 471	// �v�Z�Ŏg�p����f�[�^�̍s�� (390 - 830 nm)
#define DATA_MIN 360	// �g�p������g���̍ŏ��l
#define DATA_MAX 830	// �g�p������g���̍ő�l
#define PI 3.141592		// �~����

#define BLOCKSIZE 471		// 1�u���b�N������̃X���b�h��
#define DATANUM 50			// �v�Z���鐔
#define CALCNUM 1000		// �ׂ��悷�鐔
#define SIMNUM 100	    	// �V�~�����[�V���������
#define LOOPNUM 1			// SIMNUM��̃V�~�����[�V�����J��Ԃ���


#define GAUSS_CNT 10        // �������킹��K�E�V�A���̐�
#define GAUSS_PER 3         // �K�E�V�A���̃p�����[�^��

#define MU_MIN  360         // �ʂ̍ŏ��l
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
    ///* �W���ϑ���(CIE LMS)�̓ǂݍ��� */
    ///* �t�@�C���I�[�v�� */
    //fp_obs = fopen("./std_obs_10deg.csv", "r");
    ///* �������J���Ă��邩���`�F�b�N */
    //if (fp_obs == NULL) {
    //    cout << "File open error" << endl;
    //    return -1;
    //}

    ///* �t�@�C���ǂݍ��� */
    //for (int i = 0; i < OBS_ROW; i++) {
    //    /* 1���I�ɔg���ƃf�[�^���i�[����ϐ� */
    //    double tmp_spt = 0, tmp_l = 0, tmp_m = 0, tmp_s = 0;
    //    /* 1�s���ǂݍ��� */
    //    ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &tmp_spt, &tmp_l, &tmp_m, &tmp_s);
    //    /* �I������ */
    //    if (tmp_spt > DATA_MAX) {
    //        break;
    //    }
    //    /* �J�E���^�̍X�V */
    //    if (tmp_spt >= DATA_MIN) {
    //        obs_l[count] = tmp_l;
    //        obs_m[count] = tmp_m;
    //        obs_s[count] = tmp_s;
    //        count++;
    //    }
    //    /* �G���[�����o�����ۂ̏��� */
    //    if (ret == EOF) {
    //        cout << "error" << endl;
    //        return -1;
    //    }
    //}
    //fclose(fp_obs);
    //count = 0;
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

    /* ���ʂ��i�[����V�F�A�[�h������ */
    __shared__ double calc_data[BLOCK_SIZE][3];

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

            printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
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
    }
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
    result = new double[3 * DATANUM * CALCNUM];
    fin_result = new double[3 * SIMNUM * CALCNUM * LOOPNUM];
    mesh_result = new int[DATANUM * CALCNUM];
    mesh_f_result = new int[SIMNUM * CALCNUM * LOOPNUM];
    lms_result = new double[3 * DATANUM * CALCNUM];
    lms_fin = new double[3 * SIMNUM * CALCNUM * LOOPNUM];

    /* �t�@�C���ǂݍ��݊֐����s */
    int f_result = getFileData(d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z);

    /* �K�E�V�A���̗v�f�𐶐� */
    calcGauss(gauss_data);

    /* �]��v�Z */
    int remain = getRemain();

    /* CUDA�p�̕ϐ� */
    double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result, * d_lms;
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

    for (int i = 0; i < LOOPNUM; i++) {
        for (int j = 0; j < SIMNUM; j += DATANUM) {
            int sim_pos = (i * SIMNUM) + j;
            /* �ϕ��v�Z */
            colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > (sim_pos, d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh, (double)GAUSS_CNT, (double)DATA_MIN);
            cudaDeviceSynchronize();
            /* ���ʂ̃R�s�[ */
            cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);
            for (int k = 0; k < (3 * DATANUM * CALCNUM); k++) {
                fin_result[mem_cnt] = result[k];
                mem_cnt++;
            }
            for (int k = 0; k < (DATANUM * CALCNUM); k++) {
                mesh_f_result[mesh_cnt] = mesh_result[k];
                mesh_cnt++;
            }
        }
    }

    /* ���ʂ��I�������𖞂����Ă���Ƃ��ɒl��0�ɂ��� */
    for (int i = 0; i < LOOPNUM; i++) {
        for (int j = 0; j < SIMNUM; j++) {
            for (int k = 0; k < CALCNUM; k++) {
                int aPos = (i * 3 * SIMNUM * CALCNUM) + (j * 3 * CALCNUM) + k;
                if ((fin_result[0] * 0.005) > fin_result[aPos] &&
                    (fin_result[CALCNUM] * 0.005) > fin_result[aPos + CALCNUM] &&
                    (fin_result[CALCNUM * 2] * 0.005) > fin_result[aPos + (CALCNUM * 2)]) {
                    fin_result[aPos] = 0;
                    fin_result[aPos + CALCNUM] = 0;
                    fin_result[aPos + (CALCNUM * 2)] = 0;

                    aPos = (i * SIMNUM * CALCNUM) + (j * CALCNUM) + k;
                    mesh_f_result[aPos] = -1;
                }
            }
        }
    }

    /* �o�̓f�B���N�g�� */
    //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v1/";
    //string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_10000_10_v2/";
    string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1000_15000_10_v1/";
    
    /* �o�͂����t�@�C���̏����L�^����t�@�C�� */
    string f_info = "sim_file_info.txt";
    f_info = directory + f_info;
    ofstream o_f_info(f_info);
    /* �t�@�C���������� */
    for (int i = 0; i < LOOPNUM; i++) {
        /* �o�̓t�@�C���� */
        string fname1 = "sim_result_L_xyz_1023_";
        string fname2 = "sim_result_S_xyz_1023_";
        string fname3 = "sim_result_lms_1023_";
        string fend = ".csv";
        fname1 = directory + fname1 + to_string(i + 1) + fend;
        fname2 = directory + fname2 + to_string(i + 1) + fend;
        fname3 = directory + fname3 + to_string(i + 1) + fend;

        /* �t�@�C���o�̓X�g���[�� */
        ofstream o_file1(fname1);
        ofstream o_file2(fname2);
        ofstream o_file3(fname3);

        /* �o�͂����t�@�C���̏����L�^����t�@�C���Ƀt�@�C�������o�� */
        o_f_info << fname1 << endl;
        o_f_info << fname2 << endl;
        o_f_info << fname3 << endl;

        /* �t�@�C���ւ̏o�͌����w�� */
        o_file1 << fixed << setprecision(3);
        o_file2 << fixed << setprecision(3);
        o_file3 << fixed << setprecision(3);

        /* �l�o�� */
        for (int j = 0; j < CALCNUM; j++) {
            for (int k = 0; k < SIMNUM; k++) {
                /* �z��̗v�f�ԍ� */
                int apos = (SIMNUM * CALCNUM * 3 * i) + (CALCNUM * 3 * k) + j;
                int m_apos = (SIMNUM * CALCNUM * i) + (CALCNUM * k) + j;

                double X = fin_result[apos];
                double Y = fin_result[apos + CALCNUM];
                double Z = fin_result[apos + (CALCNUM * 2)];

                /* k ���Ō�̂Ƃ� */
                if (k == (SIMNUM - 1)) {
                    /* XYZ == 0�̂Ƃ� */
                    if (X == 0 && Y == 0 && Z == 0) {
                        o_file1 << ",,";
                        o_file2 << ",,," << mesh_f_result[m_apos];
                    }
                    /* ����ȊO�̂Ƃ� */
                    else {
                        double x = X / (X + Y + Z);
                        double y = Y / (X + Y + Z);
                        double z = Z / (X + Y + Z);

                        o_file1 << X << "," << Y << "," << Z;
                        o_file2 << x << "," << y << "," << z << "," << mesh_f_result[m_apos];
                    }
                }
                /* k���Ō�ȊO�̂Ƃ� */
                else {
                    /* XYZ == 0�̂Ƃ� */
                    if (X == 0 && Y == 0 && Z == 0) {
                        o_file1 << ",,,";
                        o_file2 << ",,," << mesh_f_result[m_apos] << ",";
                    }
                    /* ����ȊO�̂Ƃ� */
                    else {
                        double x = X / (X + Y + Z);
                        double y = Y / (X + Y + Z);
                        double z = Z / (X + Y + Z);

                        o_file1 << X << "," << Y << "," << Z << ",";
                        o_file2 << x << "," << y << "," << z << "," << mesh_f_result[m_apos] << ",";
                    }
                }
            }
            o_file1 << endl << flush;
            o_file2 << endl << flush;
            o_file3 << endl << flush;
        }
        /* �t�@�C���N���[�Y */
        o_file1.close();
        o_file2.close();
        o_file3.close();
    }
    ///* �t�@�C���o�̓X�g���[�� */
    //ofstream o_file(o_fname);

    ///* �t�@�C���o�� */
    //for (int i = 0; i < (SIMNUM * LOOPNUM * GAUSS_CNT * GAUSS_PER); i += 3) {
    //    o_file << gauss_data[i] << "," << gauss_data[i+1] << "," << gauss_data[i+2] << endl;
    //}

    /* ��������� */
    cudaFree(d_result);
    cudaFree(d_d65);
    cudaFree(d_gauss_data);
    cudaFree(d_obs_x);
    cudaFree(d_obs_y);
    cudaFree(d_obs_z);
    //cudaFree(d_lms);

    /* �z�X�g��������� */
    delete[] d65;
    delete[] obs_x;
    delete[] obs_y;
    delete[] obs_z;
    delete[] gauss_data;
    delete[] result;
    delete[] fin_result;
    delete[] lms_result;
    delete[] lms_fin;

    return 0;
}