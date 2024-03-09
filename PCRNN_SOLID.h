#pragma once

// PCRNN - Partly Connected Reflection Neural Network
// PCRNN is a new neural network structure created by Henry Z
// It is designed to resolve the problem of deficient generalization ability in fully connected neural networks
// The PCRNN is a partly connected neural network, which means that not all neurons are connected to each other
// The PCRNN is also a reflection neural network, which means that some neurons are connected to few neurons in the previous layer
// and those few neurons will tranfer values to few neurons in layer no. + 2

#include <fstream>
#include <string>
#include <iostream>
#include <unordered_set>
#include <random>
#include <memory>
#include <sstream>
#include <immintrin.h>

#define FeatureTransformer 64

// 2 = WHITE and BLACK
// 16 = 10 numbers + 6 characters
// 64 = SHA256 Value
// #define L1 (2 * 16 * 64)

// Newly Modified L1 = (WHITE and BLACK) * (1e-1 digit + 1e-2 digit) * FeatureTransformer
// Smaller L1 can improve ram usage and training & running speed
// Comparing with previous L1, there are a deficiency of features for that big L1 to extract
#define L1 (2 * 2 * FeatureTransformer)

// 2 * FeatureTransformer Size
#define L2 (2 * FeatureTransformer)

// FeatureTransformer Size
// Delicately designed for better output
#define L3 FeatureTransformer

/*
* // Two Outputs: W, B Evaluation
* // Ratio = B / W * 100%
* #define L4 2
*/

// Comparing with previous L4, this one can better fit the score offered in training dataset
#define L4 1

// Here comes some SIMD functions replaced by some easy - to - use names
// In most cases, as cpus for home are fully advanced
// therefore, in most cases, only avx512 and avx2 (a bit slower than avx512) are recommended
#if defined(__AVX512F__)
using vec_t = __m512i;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_mul_16(a, b) _mm512_mullo_epi16(a, b)
#define vec_zero() _mm512_setzero_epi32()
#define vec_set_16(a) _mm512_set1_epi16(a)
#define vec_max_16(a, b) _mm512_max_epi16(a, b)
#define vec_min_16(a, b) _mm512_min_epi16(a, b)

#elif defined(__AVX2__) || defined(__AVX__)
using vec_t = __m256i;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_mul_16(a, b) _mm256_mullo_epi16(a, b)
#define vec_zero() _mm256_setzero_si256()
#define vec_set_16(a) _mm256_set1_epi16(a)
#define vec_max_16(a, b) _mm256_max_epi16(a, b)
#define vec_min_16(a, b) _mm256_min_epi16(a, b)
#endif

namespace PCRNN
{
	int lr = 4.375e-4;
	double loss = 0;
	int thres = 1e-3;
	int losscounter = 0;
	int batch = 0;
	int batchsize = 1e7;
	double gamma = 0.995;

	int reflection_counter = 0;

	double pc_ratio = 1 / 8;
	// Vector that stores neuron index for reflection
	// Size = (FeatureTransformer + L1 + L2 + L3) * pc_ratio
	// Stucture = 8 + 32 + 16 + 8 = 64
	std::vector<int> reflection_neurons;

	std::vector<double> loss_his;

	struct Neuron
	{
		std::vector<double> value, weight, bias;
	};

	// ReLU improved by SIMD
	inline double ReLU(const double& input)
	{
		// return (std::max)((double)0, input);
		const auto in = reinterpret_cast<const vec_t*>(&input);
		const auto zero = vec_zero();
		const auto out = vec_max_16(*in, zero);
		double result;
		vec_store(reinterpret_cast<vec_t*>(&result), out);
		return result;
	}

	inline double Sigmoid(const double& input)
	{
		return 1 / (1 + exp(-input));
	}

	std::vector<Neuron> feature(FeatureTransformer);
	std::vector<Neuron> layer1(L1);
	std::vector<Neuron> layer2(L2);
	std::vector<Neuron> layer3(L3);
	std::vector<Neuron> layer4(L4);

	std::vector<double> deltaf(FeatureTransformer);
	std::vector<double> delta1(L1);
	std::vector<double> delta2(L2);
	std::vector<double> delta3(L3);
	std::vector<double> delta4(L4);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	std::uniform_int_distribution<> re_disf(0, FeatureTransformer - 1);
	std::uniform_int_distribution<> re_dis1(0, L1 - 1);
	std::uniform_int_distribution<> re_dis2(0, L2 - 1);
	std::uniform_int_distribution<> re_dis3(0, L3 - 1);

	int f_rand_offset = 0;
	int l1_rand_offset = 0;
	int l2_rand_offset = 0;

	// Values are set to zero
	// Weights and Biases are initialized randomly from 0 to 1
	void init()
	{
		for (int i = 0; i < FeatureTransformer; i++)
		{
			feature[i].value.emplace_back(0);
			feature[i].weight.emplace_back(dis(gen));
			feature[i].bias.emplace_back(dis(gen));
		}

		for (int i = 0; i < L1; i++)
		{
			for (int j = 0; j < FeatureTransformer; j++)
			{
				layer1[i].value.emplace_back(0);
				layer1[i].weight.emplace_back(dis(gen));
				layer1[i].bias.emplace_back(dis(gen));
			}
		}

		for (int i = 0; i < L2; i++)
		{
			for (int j = 0; j < L1; j++)
			{
				layer2[i].value.emplace_back(0);
				layer2[i].weight.emplace_back(dis(gen));
				layer2[i].bias.emplace_back(dis(gen));
			}
		}

		for (int i = 0; i < L3; i++)
		{
			for (int j = 0; j < L2; j++)
			{
				layer3[i].value.emplace_back(0);
				layer3[i].weight.emplace_back(dis(gen));
				layer3[i].bias.emplace_back(dis(gen));
			}
		}

		for (int i = 0; i < L4; i++)
		{
			for (int j = 0; j < L3; j++)
			{
				layer4[i].value.emplace_back(0);
				layer4[i].weight.emplace_back(dis(gen));
				layer4[i].bias.emplace_back(dis(gen));
			}
		}

		for (int i = 0; i < FeatureTransformer; i++)
		{
			if ((i + 1) * pc_ratio == 0)
			{
				reflection_neurons.emplace_back(i);
			}
		}

		for (int i = 0; i < L1; i++)
		{
			if ((i + 1) * pc_ratio == 0)
			{
				reflection_neurons.emplace_back(i);
			}
		}

		for (int i = 0; i < L2; i++)
		{
			if ((i + 1) * pc_ratio == 0)
			{
				reflection_neurons.emplace_back(i);
			}
		}

		for (int i = 0; i < L3; i++)
		{
			if ((i + 1) * pc_ratio == 0)
			{
				reflection_neurons.emplace_back(i);
			}
		}
	}

	void forward()
	{
		for (int i = 0; i < L1; i++)
		{
			for (int j = 0; j < FeatureTransformer; j++)
			{
				for (int k = 0; k < FeatureTransformer * pc_ratio; k++)
				{
					if (i != reflection_neurons[k] && j != reflection_neurons[k])
					{
						const auto val = reinterpret_cast<const vec_t*>(&feature[j].value);
						const auto w = reinterpret_cast<const vec_t*>(&layer1[i].weight[j]);
						const auto b = reinterpret_cast<const vec_t*>(&layer1[i].bias[j]);
						double result;
						vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
						layer1[i].value[j] = ReLU(result);

						for (int r = f_rand_offset; r < f_rand_offset + 4; r++)
						{
							if (r != reflection_neurons[k])
							{
								const auto valr = reinterpret_cast<const vec_t*>(&feature[r].value);
								const auto wr = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].weight[r]);
								const auto br = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].bias[r]);
								double resultr;
								vec_store(reinterpret_cast<vec_t*>(&resultr), vec_add_16(vec_mul_16(*valr, *wr), *br));
								layer1[reflection_neurons[k]].value[r] = ReLU(resultr);
							}
							else
							{
								while (r == reflection_neurons[k])
								{
									reflection_counter++;

									for (int offset = 3 - r; offset < 4 + reflection_counter; offset++)
									{
										if (offset == reflection_neurons[k])
										{
											continue;
										}

										const auto valr = reinterpret_cast<const vec_t*>(&feature[offset].value);
										const auto wr = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].weight[offset]);
										const auto br = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].bias[offset]);
										double resultr;
										vec_store(reinterpret_cast<vec_t*>(&resultr), vec_add_16(vec_mul_16(*valr, *wr), *br));
										layer1[reflection_neurons[k]].value[offset] = ReLU(resultr);
									}
								}
							}
						}
					}
				}
			}
		}

		reflection_counter = 0;

		for (int i = 0; i < L2; i++)
		{
			for (int j = 0; j < L1; j++)
			{
				for (int k = FeatureTransformer * pc_ratio; k < (FeatureTransformer + L1) * pc_ratio; k++)
				{
					if (i != reflection_neurons[k] && j != reflection_neurons[k])
					{
						const auto val = reinterpret_cast<const vec_t*>(&layer1[j].value);
						const auto w = reinterpret_cast<const vec_t*>(&layer2[i].weight[j]);
						const auto b = reinterpret_cast<const vec_t*>(&layer2[i].bias[j]);
						double result;
						vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
						layer2[i].value[j] = ReLU(result);

						if (j + l1_rand_offset % 2 == 0)
						{
							const auto val1 = reinterpret_cast<const vec_t*>(&layer1[j - 1].value);
							const auto w1 = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].weight[j - 1]);
							const auto b1 = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].bias[j - 1]);
							double result1;
							vec_store(reinterpret_cast<vec_t*>(&result1), vec_add_16(vec_mul_16(*val1, *w1), *b1));
							layer2[reflection_neurons[k]].value[j - 1] = ReLU(result1);

							const auto val2 = reinterpret_cast<const vec_t*>(&layer1[j].value);
							const auto w2 = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].weight[j]);
							const auto b2 = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].bias[j]);
							double result2;
							vec_store(reinterpret_cast<vec_t*>(&result2), vec_add_16(vec_mul_16(*val2, *w2), *b2));
							layer2[reflection_neurons[k]].value[j] = ReLU(result2);
						}
					}
					else
					{
						for (int p = 0; p < FeatureTransformer * pc_ratio; p++)
						{
							const auto valp = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[p]].value);
							const auto wp = reinterpret_cast<const vec_t*>(&feature[reflection_neurons[k]].weight[reflection_neurons[p]]);
							const auto bp = reinterpret_cast<const vec_t*>(&feature[reflection_neurons[k]].bias[reflection_neurons[p]]);
							double resultp;
							vec_store(reinterpret_cast<vec_t*>(&resultp), vec_add_16(vec_mul_16(*valp, *wp), *bp));
							feature[reflection_neurons[k]].value[reflection_neurons[p]] = ReLU(resultp);

							for (int n = (FeatureTransformer + L1) * pc_ratio ; n < (FeatureTransformer + L1 + L2) * pc_ratio; n++)
							{
								const auto valn = reinterpret_cast<const vec_t*>(&feature[reflection_neurons[p]].value);
								const auto wn = reinterpret_cast<const vec_t*>(&layer2[i].weight[reflection_neurons[p]]);
								const auto bn = reinterpret_cast<const vec_t*>(&layer2[i].bias[reflection_neurons[p]]);
								double resultn;
								vec_store(reinterpret_cast<vec_t*>(&resultn), vec_add_16(vec_mul_16(*valn, *wn), *bn));
								layer2[i].value[reflection_neurons[p]] = ReLU(resultn);
							}
						}
					}
				}
			}
		}

		reflection_counter = 0;

		for (int i = 0; i < L3; i++)
		{
			for (int j = 0; j < L2; j++)
			{
				for (int k = (FeatureTransformer + L1) * pc_ratio; k < (FeatureTransformer + L1 + L2) * pc_ratio; k++)
				{
					if (i != reflection_neurons[k] && j != reflection_neurons[k])
					{
						const auto val = reinterpret_cast<const vec_t*>(&layer2[j].value);
						const auto w = reinterpret_cast<const vec_t*>(&layer3[i].weight[j]);
						const auto b = reinterpret_cast<const vec_t*>(&layer3[i].bias[j]);
						double result;
						vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
						layer3[i].value[j] = ReLU(result);

						if (j + l2_rand_offset % 2 == 0)
						{
							const auto val1 = reinterpret_cast<const vec_t*>(&layer2[j - 1].value);
							const auto w1 = reinterpret_cast<const vec_t*>(&layer3[reflection_neurons[k]].weight[j - 1]);
							const auto b1 = reinterpret_cast<const vec_t*>(&layer3[reflection_neurons[k]].bias[j - 1]);
							double result1;
							vec_store(reinterpret_cast<vec_t*>(&result1), vec_add_16(vec_mul_16(*val1, *w1), *b1));
							layer3[reflection_neurons[k]].value[j - 1] = ReLU(result1);

							const auto val2 = reinterpret_cast<const vec_t*>(&layer2[j].value);
							const auto w2 = reinterpret_cast<const vec_t*>(&layer3[reflection_neurons[k]].weight[j]);
							const auto b2 = reinterpret_cast<const vec_t*>(&layer3[reflection_neurons[k]].bias[j]);
							double result2;
							vec_store(reinterpret_cast<vec_t*>(&result2), vec_add_16(vec_mul_16(*val2, *w2), *b2));
							layer3[reflection_neurons[k]].value[j] = ReLU(result2);
						}
					}
					else
					{
						for (int p = FeatureTransformer * pc_ratio; p < (FeatureTransformer + L1) * pc_ratio; p++)
						{
							const auto valp = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[p]].value);
							const auto wp = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].weight[reflection_neurons[p]]);
							const auto bp = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[k]].bias[reflection_neurons[p]]);
							double resultp;
							vec_store(reinterpret_cast<vec_t*>(&resultp), vec_add_16(vec_mul_16(*valp, *wp), *bp));
							layer1[reflection_neurons[k]].value[reflection_neurons[p]] = ReLU(resultp);

							for (int n = (FeatureTransformer + L1 + L2) * pc_ratio; n < (FeatureTransformer + L1 + L2 + L3) * pc_ratio; n++)
							{
								const auto valn = reinterpret_cast<const vec_t*>(&layer1[reflection_neurons[p]].value);
								const auto wn = reinterpret_cast<const vec_t*>(&layer3[i].weight[reflection_neurons[p]]);
								const auto bn = reinterpret_cast<const vec_t*>(&layer3[i].bias[reflection_neurons[p]]);
								double resultn;
								vec_store(reinterpret_cast<vec_t*>(&resultn), vec_add_16(vec_mul_16(*valn, *wn), *bn));
								layer3[i].value[reflection_neurons[p]] = ReLU(resultn);
							}
						}
					}
				}
			}
		}

		reflection_counter = 0;

		for (int i = 0; i < L4; i++)
		{
			for (int j = 0; j < L3; j++)
			{
				for (int k = (FeatureTransformer + L1 + L2) * pc_ratio; k < (FeatureTransformer + L1 + L2 + L3) * pc_ratio; k++)
				{
					if (i != reflection_neurons[k] && j != reflection_neurons[k])
					{
						const auto val = reinterpret_cast<const vec_t*>(&layer3[j].value);
						const auto w = reinterpret_cast<const vec_t*>(&layer4[i].weight[j]);
						const auto b = reinterpret_cast<const vec_t*>(&layer4[i].bias[j]);
						double result;
						vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
						layer4[i].value[j] = Sigmoid(result);
					}
					else
					{
						for (int p = (FeatureTransformer + L1) * pc_ratio; p < (FeatureTransformer + L1 + L2) * pc_ratio; p++)
						{
							const auto valp = reinterpret_cast<const vec_t*>(&layer3[reflection_neurons[p]].value);
							const auto wp = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].weight[reflection_neurons[p]]);
							const auto bp = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[k]].bias[reflection_neurons[p]]);
							double resultp;
							vec_store(reinterpret_cast<vec_t*>(&resultp), vec_add_16(vec_mul_16(*valp, *wp), *bp));
							layer2[reflection_neurons[k]].value[reflection_neurons[p]] = ReLU(resultp);

							for (int n = (FeatureTransformer + L1 + L2 + L3) * pc_ratio; n < (FeatureTransformer + L1 + L2 + L3 + L4) * pc_ratio; n++)
							{
								const auto valn = reinterpret_cast<const vec_t*>(&layer2[reflection_neurons[p]].value);
								const auto wn = reinterpret_cast<const vec_t*>(&layer4[i].weight[reflection_neurons[p]]);
								const auto bn = reinterpret_cast<const vec_t*>(&layer4[i].bias[reflection_neurons[p]]);
								double resultn;
								vec_store(reinterpret_cast<vec_t*>(&resultn), vec_add_16(vec_mul_16(*valn, *wn), *bn));
								layer4[i].value[reflection_neurons[p]] = Sigmoid(resultn);
							}
						}
					}
				}
			}
		}

		reflection_counter = 0;

	}

	void backward(const int& score)
	{
		for (int i = 0; i < L4; i++)
		{
			for (int j = 0; j < L3; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer3[j].value);
				const auto w = reinterpret_cast<const vec_t*>(&layer4[i].weight[j]);
				const auto b = reinterpret_cast<const vec_t*>(&layer4[i].bias[j]);
				double result;
				vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
				delta4[i] += -(score - result) * result * (1.0 - result);
			}
		}

		for (int i = 0; i < L3; i++)
		{
			for (int j = 0; j < L2; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer2[j].value);
				const auto w = reinterpret_cast<const vec_t*>(&layer3[i].weight[j]);
				const auto b = reinterpret_cast<const vec_t*>(&layer3[i].bias[j]);
				double result;
				vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
				delta3[i] += -(score - result) * result * (1.0 - result);
			}
		}

		for (int i = 0; i < L2; i++)
		{
			for (int j = 0; j < L1; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer1[j].value);
				const auto w = reinterpret_cast<const vec_t*>(&layer2[i].weight[j]);
				const auto b = reinterpret_cast<const vec_t*>(&layer2[i].bias[j]);
				double result;
				vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
				delta2[i] += -(score - result) * result * (1.0 - result);
			}
		}

		for (int i = 0; i < L1; i++)
		{
			for (int j = 0; j < FeatureTransformer; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&feature[j].value);
				const auto w = reinterpret_cast<const vec_t*>(&layer1[i].weight[j]);
				const auto b = reinterpret_cast<const vec_t*>(&layer1[i].bias[j]);
				double result;
				vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
				delta1[i] += -(score - result) * result * (1.0 - result);
			}
		}

		for (int i = 0; i < L4; i++)
		{
			for (int j = 0; j < L3; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer3[j].value);
				const auto delta = reinterpret_cast<const vec_t*>(&delta4[i]);
				const auto LR = reinterpret_cast<const vec_t*>(&lr);
				double fweight;
				double fbias;
				vec_store(reinterpret_cast<vec_t*>(&fweight), vec_mul_16(*LR, vec_mul_16(*delta, *val)));
				vec_store(reinterpret_cast<vec_t*>(&fbias), vec_mul_16(*LR, *delta));
				layer4[i].weight[j] -= fweight;
				layer4[i].bias[j] -= fbias;
			}
		}

		for (int i = 0; i < L3; i++)
		{
			for (int j = 0; j < L2; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer2[j].value);
				const auto delta = reinterpret_cast<const vec_t*>(&delta3[i]);
				const auto LR = reinterpret_cast<const vec_t*>(&lr);
				double fweight;
				double fbias;
				vec_store(reinterpret_cast<vec_t*>(&fweight), vec_mul_16(*LR, vec_mul_16(*delta, *val)));
				vec_store(reinterpret_cast<vec_t*>(&fbias), vec_mul_16(*LR, *delta));
				layer3[i].weight[j] -= fweight;
				layer3[i].bias[j] -= fbias;
			}
		}

		for (int i = 0; i < L2; i++)
		{
			for (int j = 0; j < L1; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&layer1[j].value);
				const auto delta = reinterpret_cast<const vec_t*>(&delta2[i]);
				const auto LR = reinterpret_cast<const vec_t*>(&lr);
				double fweight;
				double fbias;
				vec_store(reinterpret_cast<vec_t*>(&fweight), vec_mul_16(*LR, vec_mul_16(*delta, *val)));
				vec_store(reinterpret_cast<vec_t*>(&fbias), vec_mul_16(*LR, *delta));
				layer2[i].weight[j] -= fweight;
				layer2[i].bias[j] -= fbias;
			}
		}

		for (int i = 0; i < L1; i++)
		{
			for (int j = 0; j < FeatureTransformer; j++)
			{
				const auto val = reinterpret_cast<const vec_t*>(&feature[j].value);
				const auto delta = reinterpret_cast<const vec_t*>(&delta1[i]);
				const auto LR = reinterpret_cast<const vec_t*>(&lr);
				double fweight;
				double fbias;
				vec_store(reinterpret_cast<vec_t*>(&fweight), vec_mul_16(*LR, vec_mul_16(*delta, *val)));
				vec_store(reinterpret_cast<vec_t*>(&fbias), vec_mul_16(*LR, *delta));
				layer1[i].weight[j] -= fweight;
				layer1[i].bias[j] -= fbias;
			}
		}
	}

	double Loss(const double& score)
	{
		for (int i = 0; i < L4; i++)
		{
			loss += (layer4[i].value[0] - score) * (layer4[i].value[0] - score) / 2;
		}

		return loss;
	}

	void saveNN(const std::string& filepath)
	{
		std::ofstream nnfile(filepath.c_str());

		nnfile << FeatureTransformer << " " << L1 << " " << L2 << " " << L3 << " " << L4 << std::endl;

		for (int i = 0; i < FeatureTransformer; i++)
		{
			nnfile << feature[i].weight[0] << " " << feature[i].bias[0] << std::endl;
		}

		for (int i = 0; i < L1; i++)
		{
			for (int j = 0; j < FeatureTransformer; j++)
			{
				nnfile << layer1[i].weight[j] << " " << layer1[i].bias[j] << std::endl;
			}
		}

		for (int i = 0; i < L2; i++)
		{
			for (int j = 0; j < L1; j++)
			{
				nnfile << layer2[i].weight[j] << " " << layer2[i].bias[j] << std::endl;
			}
		}

		for (int i = 0; i < L3; i++)
		{
			for (int j = 0; j < L2; j++)
			{
				nnfile << layer3[i].weight[j] << " " << layer3[i].bias[j] << std::endl;
			}
		}

		for (int i = 0; i < L4; i++)
		{
			for (int j = 0; j < L3; j++)
			{
				nnfile << layer4[i].weight[j] << " " << layer4[i].bias[j] << std::endl;
			}
		}

		nnfile << "pcr" << " " << pc_ratio << std::endl;
		nnfile << "fro" << " " << f_rand_offset << std::endl;
		nnfile << "l1ro" << " " << l1_rand_offset << std::endl;
		nnfile << "l2ro" << " " << l2_rand_offset << std::endl;

		nnfile.close();
	}

	bool trend(std::vector<double> &history)
	{
		double sum_x2 = 0.0;
		double sum_y = 0.0;
		double sum_x = 0.0;
		double sum_xy = 0.0;
		
		double a = 0.0;
		double b = 0.0;

		for (int i = 0; i < history.size(); i++)
		{
			sum_x2 += i * i;
			sum_y += history[i];
			sum_x += i;
			sum_xy += i * history[i];
		}

		double tmp = history.size() * sum_x2 - sum_x * sum_x;

		if (std::abs(tmp) > 0.000001f)
		{
			a = (history.size() * sum_xy - sum_x * sum_y) / tmp;
			b = (sum_y * sum_x2 - sum_x * sum_xy) / tmp;
		}

		if (a <= -0.3)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void tune_reflection()
	{
		f_rand_offset = re_disf(gen);
		l1_rand_offset = re_dis1(gen);
		l2_rand_offset = re_dis2(gen);

		for (int i = 0; i < FeatureTransformer * pc_ratio; i++)
		{
			reflection_neurons[i] = re_disf(gen);
			
			if (reflection_neurons[i] != i)
			{
				std::swap(feature[re_disf(gen)], feature[reflection_neurons[i]]);
			}
		}

		for (int i = 0; i < L1 * pc_ratio; i++)
		{
			reflection_neurons[i] = re_dis1(gen);

			if (reflection_neurons[i] != i)
			{
				std::swap(layer1[re_dis1(gen)], layer1[reflection_neurons[i]]);
			}
		}

		for (int i = 0; i < L2 * pc_ratio; i++)
		{
			reflection_neurons[i] = re_dis2(gen);

			if (reflection_neurons[i] != i)
			{
				std::swap(layer2[re_dis2(gen)], layer2[reflection_neurons[i]]);
			}
		}

		for (int i = 0; i < L3 * pc_ratio; i++)
		{
			reflection_neurons[i] = re_dis3(gen);

			if (reflection_neurons[i] != i)
			{
				std::swap(layer3[re_dis3(gen)], layer3[reflection_neurons[i]]);
			}
		}
	}

	void train(std::vector<double> element, const double& score)
	{
		for (int i = 0; i < FeatureTransformer; i++)
		{
			feature[i].value.emplace_back(element[i]);
		}

		forward();

		if (batch % batchsize == 0)
		{
			lr *= gamma;
		}

		std::cout << "Loss: " << Loss(score) << std::endl;

		if (Loss(score) < thres)
		{
			losscounter++;

			if (losscounter == 50)
			{
				saveNN("C:\\Users\\Henry Z\\Desktop\\VI\\Visual Studio\\PCRNN\\x64\\Release\\PCRNN.nn");
				exit(0);
			}
		}

		backward(score);

		if (loss_his.size() == 50)
		{
			if (trend(loss_his))
			{
				tune_reflection();
			}

			loss_his.clear();
		}
		else
		{
			loss_his.emplace_back(Loss(score));
		}
	}
}