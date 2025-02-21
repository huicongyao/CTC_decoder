#pragma once

#include <iostream>
#include <torch/torch.h>
#include <cstring>
#include <cmath>

uint8_t get_qual(float x);
void ctc_greedy_decode(float * inputs, uint8_t *seqs, uint8_t * moves, uint8_t * quals, int T, int N, int C) ;