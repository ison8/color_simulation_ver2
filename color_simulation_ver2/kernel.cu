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
#define SIMNUM 1023			// �V�~�����[�V���������
#define LOOPNUM 10			// SIMNUM��̃V�~�����[�V�����J��Ԃ���

using namespace std;