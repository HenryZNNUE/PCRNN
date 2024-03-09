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

	std::vector<int> layer_structure;
	std::vector<Neuron> layer;
	std::vector<double> delta;

	std::random_device rd;
	std::mt19937 gen(rd());

	int rand_offset = 0;

	void set_PCRNN_structure(std::vector<int> structure)
	{
		for (int i = 0; i < structure.size(); i++)
		{
			layer_structure.emplace_back(structure[i]);
		}
	}

	// Values are set to zero
	// Weights and Biases are initialized randomly from 0 to 1
	void init()
	{
		for (int i = 0; i < layer_structure.size(); i++)
		{
			for (int i = 0; i < layer_structure[i]; i++)
			{
				layer[i].value.emplace_back(0);
				layer[i].weight.emplace_back(dis(gen));
				layer[i].bias.emplace_back(dis(gen));

				if ((i + 1) * pc_ratio == 0)
				{
					reflection_neurons.emplace_back(i);
				}
			}
		}
	}

	void forward()
	{
		for (int i = 0; i < layer_structure.size(); i++)
		{
			for (int j = layer_structure[i]; j < layer_structure[(((i + 1) == layer_structure.size()) ? (layer_structure.size() - 1) : (i + 1))]; j++)
			{
				for (int k = ((i == 0) ? 0 : layer_structure[i - 1]); k < layer_structure[i]; k++)
				{
					for (int l = ((i == 0) ? 0 : layer_structure[i - 1] * pc_ratio); l < layer_structure[i] * pc_ratio; l++)
					{
						if (j != reflection_neurons[l] && k != reflection_neurons[l])
						{
							const auto val = reinterpret_cast<const vec_t*>(&layer[k].value);
							const auto w = reinterpret_cast<const vec_t*>(&layer[j].weight[k]);
							const auto b = reinterpret_cast<const vec_t*>(&layer[j].bias[k]);
							double result;
							vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
							layer[j].value[k] = ReLU(result);

							if ((k + rand_offset) % 2 == 0)
							{
								const auto val1 = reinterpret_cast<const vec_t*>(&layer[(((k - 1) < 0) ? 0 : (k - 1))].value);
								const auto w1 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[(((k - 1) < 0) ? 0 : (k - 1))]);
								const auto b1 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[(((k - 1) < 0) ? 0 : (k - 1))]);
								double result1;
								vec_store(reinterpret_cast<vec_t*>(&result1), vec_add_16(vec_mul_16(*val1, *w1), *b1));
								layer[reflection_neurons[l]].value[(((k - 1) < 0) ? 0 : (k - 1))] = ReLU(result1);

								const auto val2 = reinterpret_cast<const vec_t*>(&layer[j].value);
								const auto w2 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[j]);
								const auto b2 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[j]);
								double result2;
								vec_store(reinterpret_cast<vec_t*>(&result2), vec_add_16(vec_mul_16(*val2, *w2), *b2));
								layer[reflection_neurons[l]].value[k] = ReLU(result2);
							}
						}
						else
						{
							if (i > 0)
							{
								for (int p = ((i == 1) ? 0 : layer_structure[i - 2] * pc_ratio); p < layer_structure[i - 1] * pc_ratio; p++)
								{
									const auto valp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[p]].value);
									const auto wp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[reflection_neurons[p]]);
									const auto bp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[reflection_neurons[p]]);
									double resultp;
									vec_store(reinterpret_cast<vec_t*>(&resultp), vec_add_16(vec_mul_16(*valp, *wp), *bp));
									layer[reflection_neurons[l]].value[reflection_neurons[p]] = ReLU(resultp);

									for (int n = layer_structure[i] * pc_ratio; n < layer_structure[i + 1] * pc_ratio; n++)
									{
										const auto valn = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[p]].value);
										const auto wn = reinterpret_cast<const vec_t*>(&layer[k].weight[reflection_neurons[p]]);
										const auto bn = reinterpret_cast<const vec_t*>(&layer[k].bias[reflection_neurons[p]]);
										double resultn;
										vec_store(reinterpret_cast<vec_t*>(&resultn), vec_add_16(vec_mul_16(*valn, *wn), *bn));
										layer[j].value[reflection_neurons[p]] = ReLU(resultn);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	void backward(const int& score)
	{
		for (int i = 0; i < layer_structure.size(); i++)
		{
			for (int j = layer_structure[i]; j < layer_structure[(((i + 1) == layer_structure.size()) ? (layer_structure.size() - 1) : (i + 1))]; j++)
			{
				for (int k = ((i == 0) ? 0 : layer_structure[i - 1]); k < layer_structure[i]; k++)
				{
					for (int l = ((i == 0) ? 0 : layer_structure[i - 1] * pc_ratio); l < layer_structure[i] * pc_ratio; l++)
					{
						if (j != reflection_neurons[l] && k != reflection_neurons[l])
						{
							const auto val = reinterpret_cast<const vec_t*>(&layer[k].value);
							const auto w = reinterpret_cast<const vec_t*>(&layer[j].weight[k]);
							const auto b = reinterpret_cast<const vec_t*>(&layer[j].bias[k]);
							double result;
							vec_store(reinterpret_cast<vec_t*>(&result), vec_add_16(vec_mul_16(*val, *w), *b));
							delta[j] += -(score - result) * result * (1.0 - result);

							const auto vald = reinterpret_cast<const vec_t*>(&layer[k].value);
							const auto deltad = reinterpret_cast<const vec_t*>(&delta[j]);
							const auto LR = reinterpret_cast<const vec_t*>(&lr);
							double weightd;
							double biasd;
							vec_store(reinterpret_cast<vec_t*>(&weightd), vec_mul_16(*LR, vec_mul_16(*deltad, *vald)));
							vec_store(reinterpret_cast<vec_t*>(&biasd), vec_mul_16(*LR, *deltad));
							layer[j].weight[k] -= weightd;
							layer[j].bias[k] -= biasd;

							if (k + rand_offset % 2 == 0)
							{
								const auto val1 = reinterpret_cast<const vec_t*>(&layer[(((k - 1) < 0) ? 0 : (k - 1))].value);
								const auto w1 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[(((k - 1) < 0) ? 0 : (k - 1))]);
								const auto b1 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[(((k - 1) < 0) ? 0 : (k - 1))]);
								double result1;
								vec_store(reinterpret_cast<vec_t*>(&result1), vec_add_16(vec_mul_16(*val1, *w1), *b1));
								delta[reflection_neurons[l]] += -(score - result) * result * (1.0 - result);

								const auto vald1 = reinterpret_cast<const vec_t*>(&layer[k].value);
								const auto deltad1 = reinterpret_cast<const vec_t*>(&delta[j]);
								const auto LR1 = reinterpret_cast<const vec_t*>(&lr);
								double weightd1;
								double biasd1;
								vec_store(reinterpret_cast<vec_t*>(&weightd1), vec_mul_16(*LR1, vec_mul_16(*deltad1, *vald)));
								vec_store(reinterpret_cast<vec_t*>(&biasd1), vec_mul_16(*LR1, *deltad1));
								layer[j].weight[k] -= weightd1;
								layer[j].bias[k] -= biasd1;

								const auto val2 = reinterpret_cast<const vec_t*>(&layer[j].value);
								const auto w2 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[j]);
								const auto b2 = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[j]);
								double result2;
								vec_store(reinterpret_cast<vec_t*>(&result2), vec_add_16(vec_mul_16(*val2, *w2), *b2));
								delta[reflection_neurons[l]] += -(score - result) * result * (1.0 - result);

								const auto vald2 = reinterpret_cast<const vec_t*>(&layer[k].value);
								const auto deltad2 = reinterpret_cast<const vec_t*>(&delta[j]);
								const auto LR2 = reinterpret_cast<const vec_t*>(&lr);
								double weightd2;
								double biasd2;
								vec_store(reinterpret_cast<vec_t*>(&weightd2), vec_mul_16(*LR2, vec_mul_16(*deltad2, *vald)));
								vec_store(reinterpret_cast<vec_t*>(&biasd2), vec_mul_16(*LR2, *deltad2));
								layer[j].weight[k] -= weightd2;
								layer[j].bias[k] -= biasd2;
							}
						}
						else
						{
							for (int p = ((i == 1) ? 0 : layer_structure[i - 2] * pc_ratio); p < layer_structure[i - 1] * pc_ratio; p++)
							{
								const auto valp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[p]].value);
								const auto wp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].weight[reflection_neurons[p]]);
								const auto bp = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[l]].bias[reflection_neurons[p]]);
								double resultp;
								vec_store(reinterpret_cast<vec_t*>(&resultp), vec_add_16(vec_mul_16(*valp, *wp), *bp));
								delta[reflection_neurons[l]] += -(score - resultp) * resultp * (1.0 - resultp);

								const auto valpd = reinterpret_cast<const vec_t*>(&layer[k].value);
								const auto deltapd = reinterpret_cast<const vec_t*>(&delta[j]);
								const auto LRp = reinterpret_cast<const vec_t*>(&lr);
								double weightpd;
								double biaspd;
								vec_store(reinterpret_cast<vec_t*>(&weightpd), vec_mul_16(*LRp, vec_mul_16(*deltapd, *valpd)));
								vec_store(reinterpret_cast<vec_t*>(&biaspd), vec_mul_16(*LRp, *deltapd));
								layer[j].weight[k] -= weightpd;
								layer[j].bias[k] -= biaspd;

								for (int n = layer_structure[i] * pc_ratio; n < layer_structure[i + 1] * pc_ratio; n++)
								{
									const auto valn = reinterpret_cast<const vec_t*>(&layer[reflection_neurons[p]].value);
									const auto wn = reinterpret_cast<const vec_t*>(&layer[k].weight[reflection_neurons[p]]);
									const auto bn = reinterpret_cast<const vec_t*>(&layer[k].bias[reflection_neurons[p]]);
									double resultn;
									vec_store(reinterpret_cast<vec_t*>(&resultn), vec_add_16(vec_mul_16(*valn, *wn), *bn));
									delta[reflection_neurons[l]] += -(score - resultn) * resultn * (1.0 - resultn);

									const auto valnd = reinterpret_cast<const vec_t*>(&layer[k].value);
									const auto deltand = reinterpret_cast<const vec_t*>(&delta[j]);
									const auto LRn = reinterpret_cast<const vec_t*>(&lr);
									double weightnd;
									double biasnd;
									vec_store(reinterpret_cast<vec_t*>(&weightnd), vec_mul_16(*LRn, vec_mul_16(*deltand, *valnd)));
									vec_store(reinterpret_cast<vec_t*>(&biasnd), vec_mul_16(*LRn, *deltand));
									layer[j].weight[k] -= weightnd;
									layer[j].bias[k] -= biasnd;
								}
							}
						}
					}
				}
			}
		}
	}

	double Loss(const double& score)
	{
		for (int i = layer_structure[layer_structure.size() - 1]; i < layer_structure[layer_structure.size()]; i++)
		{
			loss += (layer[i].value[0] - score) * (layer[i].value[0] - score) / 2;
		}

		return loss;
	}

	void saveNN(const std::string& filepath)
	{
		std::ofstream nnfile(filepath.c_str());

		for (int i = 0; i < layer_structure.size(); i++)
		{
			nnfile << layer_structure[i] << " ";
		}

		nnfile << std::endl;

		for (int i = 0; i < layer_structure.size(); i++)
		{
			for (int j = layer_structure[i]; j < layer_structure[(((i + 1) == layer_structure.size()) ? (layer_structure.size() - 1) : (i + 1))]; j++)
			{
				for (int k = ((i == 0) ? 0 : layer_structure[i - 1]); k < layer_structure[i]; k++)
				{
					nnfile << layer[j].weight[k] << " " << layer[j].bias[k] << std::endl;
				}
			}
		}

		nnfile << "pcr" << " " << pc_ratio << std::endl;
		nnfile << "ro" << " " << rand_offset << std::endl;

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
		std::uniform_int_distribution<>re_dis_offset(2, 10);
		rand_offset = re_dis_offset(gen);

		for (int i = 0; i < layer_structure.size(); i++)
		{
			for (int j = layer_structure[i]; j < layer_structure[(((i + 1) == layer_structure.size()) ? (layer_structure.size() - 1) : (i + 1))] * pc_ratio; j++)
			{
				for (int k = ((i == 0) ? 0 : layer_structure[i - 1]); k < layer_structure[i] * pc_ratio; k++)
				{
					std::uniform_int_distribution<> re_dis1(0, layer_structure[(((i + 1) == layer_structure.size()) ? (layer_structure.size() - 1) : (i + 1))] - 1);
					std::uniform_int_distribution<> re_dis2(0, layer_structure[i] - 1);

					reflection_neurons[j] = re_dis1(gen);

					if (reflection_neurons[j] != j)
					{
						std::swap(layer[re_dis1(gen)], layer[reflection_neurons[j]]);
					}

					reflection_neurons[k] = re_dis2(gen);

					if (reflection_neurons[k] != k)
					{
						std::swap(layer[re_dis2(gen)], layer[reflection_neurons[k]]);
					}
				}
			}
		}
	}

	void train(std::vector<double> element, const double& score)
	{
		for (int i = 0; i < layer_structure[0]; i++)
		{
			layer[i].value.emplace_back(element[i]);
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