#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <cstddef>
#include <utility>
#include <assert.h>
#include <memory>
#include <vector>
#include <stdio.h>
#include <string.h>
#include "../errorchecks.cuh"
#include <random> // The header for the generators.

#ifdef __VERBOSE__
#define __DEBUG__
#define IF_VERBOSE(x) x
#else
#define IF_VERBOSE(x) // nothing
#endif
#ifdef __DEBUG__
extern size_t allocedBytes;
// static size_t freedBytes = 0;
extern size_t allocedTensors;
extern size_t freedTensors;
#define IF_DEBUG(x) x
#else
#define IF_DEBUG(x) // nothing
#endif

template<typename T, size_t RANK>
class Tensor {
private:
	IF_DEBUG(size_t _UID = 0);
	T* data;
	size_t sizes[RANK];
	size_t size;
	size_t *ref_counter = nullptr;
	template<typename TT, size_t RRANK> friend class Tensor; // fix other Tensors with different rank cannot access private members 
	// Tensor(T* otherData, size_t otherSize, size_t* otherRef_counter, std::initializer_list<size_t> sizes) // private reshape constructor
	// { // it is not possible to call a generic constructor
	// 	assert(sizes.size() == RANK);
	// 	this->size = 1;
	// 	auto iter = sizes.begin();
	// 	for (size_t i = 0; i < RANK; i++, iter++)
	// 	{
	// 		this->sizes[i] = *iter;
	// 		this->size *= this->sizes[i];
	// 	}

	// 	assert(this->size == otherSize);

	// 	this->data = otherData;
	// 	this->ref_counter = otherRef_counter;
	// 	(*ref_counter)++;
	// }
public:
	static Tensor<T, RANK> BuildTensor(std::vector<Tensor<T, RANK-1>>& others) {

		assert(others.size() > 0);
		std::vector<size_t> newTSizes(RANK);
		newTSizes[0] = others.size();
		for (size_t i = 0; i < RANK - 1; i++)
		{
			newTSizes[i+1] = others[0].sizes[i];
		}
		
		// check if all have same size
		for (size_t i = 0; i < others.size(); i++)
		{
			assert(others[i].size == others[0].size);
			for (size_t j = 0; j < RANK - 1; j++)
			{
				assert(others[i].sizes[j] == others[0].sizes[j]);
			}
		}

		Tensor<T, RANK> newTensor(newTSizes);
		
		// copy data
		for (size_t i = 0; i < others.size(); i++)
		{
			for (size_t j = 0; j < others[0].size; j++)
			{
				newTensor.data[i * others[0].size + j] = others[i].data[j];
			}
		}
		return newTensor;
	}
	static Tensor<T, RANK> BuildTensor(std::initializer_list<Tensor<T, RANK-1>> others) {
		std::vector<Tensor<T, RANK-1>> vec;
		auto iter = others.begin();
		while (iter != others.end()){
			vec.push_back(*iter);
			iter++;
		}
		return Tensor<T, RANK>::BuildTensor(vec);
	}
	// duplicate constructor that uses vectors instead of initializer_list
	Tensor(std::vector<size_t>& sizes) {
		assert(sizes.size() == RANK);
		this->size = 1;
		auto iter = sizes.begin();
		for (size_t i = 0; i < RANK; i++, iter++)
		{
			this->sizes[i] = *iter;
			this->size *= this->sizes[i];
		}
		// std::cout << "alloc" << std::endl;
		CHECK_CUDA_ERROR(cudaMallocManaged((void**)&data, this->size * sizeof(T)));
		IF_DEBUG(allocedBytes += this->size * sizeof(T));
		IF_DEBUG(allocedTensors++);
		IF_DEBUG(_UID = allocedTensors);
		// std::cout << data[0] << ", " << std::endl;
		ref_counter = new size_t(1);
		IF_DEBUG(std::cout << "alloc vec: \t[" << this->_UID <<", "<< this->size * sizeof(T)<< " Bytes], alloced: "<< allocedBytes<< std::endl);

	}
	Tensor(std::initializer_list<size_t> sizes) {
		assert(sizes.size() == RANK);
		this->size = 1;
		auto iter = sizes.begin();
		for (size_t i = 0; i < RANK; i++, iter++)
		{
			this->sizes[i] = *iter;
			this->size *= this->sizes[i];
		}
		// std::cout << "alloc" << std::endl;
		CHECK_CUDA_ERROR(cudaMallocManaged((void**)&data, this->size * sizeof(T)));
		IF_DEBUG(allocedBytes += this->size * sizeof(T));
		IF_DEBUG(allocedTensors++);
		IF_DEBUG(_UID = allocedTensors);
		ref_counter = new size_t(1);
		IF_DEBUG(std::cout << "alloc list: \t[" << this->_UID <<", "<< this->size * sizeof(T)<< " Bytes], alloced: "<< allocedBytes<< std::endl);
			
	}
	Tensor(const Tensor<T, RANK>& other) // copy constructor
	{
		IF_VERBOSE(std::cout << "copy constr ["<<other._UID<<" -> "<< this->_UID << "]" << std::endl);
		this->size = other.size;
		memcpy(this->sizes, other.sizes, sizeof(size_t) * RANK);
		this->data = other.data;
		this->ref_counter = other.ref_counter;
		IF_DEBUG(this->_UID = other._UID);
		(* ref_counter)++;
	}
	Tensor(Tensor<T, RANK>&& other) noexcept // move constructor
	{
		IF_VERBOSE(std::cout << "move constr ["<<other._UID<<" -> "<< this->_UID << "]" << std::endl);
		this->data = std::exchange(other.data, nullptr);
		memcpy(this->sizes, other.sizes, sizeof(size_t) * RANK);
		this->size = std::exchange(other.size, 0);
		this->ref_counter = std::exchange(other.ref_counter, nullptr);
		IF_DEBUG(this->_UID = std::exchange(other._UID, 0));
	}
	// Tensor<T, RANK>& operator=(const Tensor<T, RANK>& other) = delete;
	// Tensor<T, RANK>& operator=(Tensor<T, RANK>&& other) noexcept = delete;
	Tensor<T, RANK>& operator=(const Tensor<T, RANK>& other) // copy assignment
	{
		IF_VERBOSE(std::cout << "copy assign ["<<other._UID<<" -> "<< this->_UID << "]" << std::endl);
		Tensor<T, RANK> tmp(*this);
		*this = Tensor<T, RANK>(other);
		// (*(tmp.ref_counter))--; // TODO WHY NOT?????
		// std::cout << "ref_counter: " << (*tmp.ref_counter) << ", ID: "<< tmp._UID << std::endl;
		return *this;
	}
	Tensor<T, RANK>& operator=(Tensor<T, RANK>&& other) noexcept // move assignment
	{
		IF_VERBOSE(std::cout << "move assign ["<<other._UID<<" -> "<< this->_UID << "]" << std::endl);
		Tensor<T, RANK> tmp(*this);
		
		(*tmp.ref_counter)--;
		this->data = std::exchange(other.data, nullptr);
		memcpy(this->sizes, other.sizes, sizeof(size_t) * RANK);
		this->size = std::exchange(other.size, 0);
		this->ref_counter = std::exchange(other.ref_counter, nullptr);
		IF_DEBUG(this->_UID = std::exchange(other._UID, 0));
		return *this;
	}

	~Tensor() {
		IF_VERBOSE(std::cout << "destruct: \t[" << this->_UID <<", "<< this->size * sizeof(T) << " Bytes], remaining refs: ");
		IF_VERBOSE(if(ref_counter)std::cout <<  (*ref_counter) - 1 <<std::endl);
		IF_VERBOSE(else std::cout << "nullptr" <<std::endl);
		if (ref_counter && !(--(*ref_counter))) {
			//std::cout<<(*ref_counter)<<std::endl;
			// std::cout << "free Tensor_UID: "<<_UID << std::endl;
			IF_DEBUG(std::cout << "freeing: \t[" << this->_UID <<", "<< this->size * sizeof(T) << " Bytes], alloced: "<< allocedBytes<< " -> " << allocedBytes - this->size * sizeof(T)  << std::endl);
			
			IF_DEBUG(assert(allocedBytes >= (this->size * sizeof(T))));
			// std::cout << "ref_counter: " << (*ref_counter) << std::endl;
			// std::cout << "allocedBytes: " << allocedBytes << " Bytes = "<<allocedBytes/(1024.0*1024*1024) << "GiB"<<std::endl;
			// std::cout << "freeing: " << this->size <<" Bytes" << std::endl;
			CHECK_CUDA_ERROR(cudaFree(data));
			IF_DEBUG(allocedBytes -= this->size * sizeof(T));
			IF_DEBUG(freedTensors++);
			// cudaPointerAttributes a;
			// cudaPointerGetAttributes(&a,data);
			// std::cout<<"Memory Type :"<<a.type<<std::endl;
			// if(!(a.type==0))
			// 	CHECK_CUDA_ERROR(cudaFree(data));
			// std::cout << "free" << std::endl;
			// std::cout << "allocedBytes: " << allocedBytes << " Bytes = "<<allocedBytes/(1024.0*1024*1024) << "GiB"<<std::endl;
			delete ref_counter;
			ref_counter = nullptr;
			// (*ref_counter)--;
		}
	}

	/*
		TODO: check if cudaMemset works for managed memory
	*/
	void setZero() {
		int n = this->size;
		CHECK_CUDA_ERROR(cudaMemset(this->data, 0, n * sizeof(T)));
	}

	/*
	Resulting Tensor is alloced on the gpu and cpu using cudaMallocManaged.
	*/
	void* operator new(size_t size)
	{
		//std::cout << "Overloading new operator with size: " << size << std::endl;
		void* p;
		CHECK_CUDA_ERROR(cudaMallocManaged((void**)&p, size));
		return p;
	}
	void operator delete(void* p)
	{
		//std::cout << "Overloading delete operator " << std::endl;
		CHECK_CUDA_ERROR(cudaFree(p));
	}
	__host__ __device__ T operator [] (size_t i) const { return this->data[i]; }
	__host__ __device__ T& operator [] (size_t i) { return this->data[i]; }

	__host__ __device__ size_t pos(size_t x0) const {  return x0; }
	__host__ __device__ size_t pos(size_t x0, size_t x1) const { static_assert(RANK == 2, ""); return x0 * this->sizes[1] + x1; }
	__host__ __device__ size_t pos(size_t x0, size_t x1, size_t x2) const { static_assert(RANK == 3, ""); return x0 * this->sizes[1] * this->sizes[2] + x1 * this->sizes[2] + x2; }
	__host__ __device__ size_t pos(size_t x0, size_t x1, size_t x2, size_t x3) const { static_assert(RANK == 4, ""); return x0 * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[3] * this->sizes[2] + x2 * this->sizes[3] + x3; }
	__host__ __device__ size_t pos(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4) const { static_assert(RANK == 5, ""); return x0* this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[4] * this->sizes[3] + x3 * this->sizes[4] + x4; }
	__host__ __device__ size_t pos(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4, size_t x5) const { static_assert(RANK == 6, ""); return x0 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[5] * this->sizes[4] * this->sizes[3] + x3 * this->sizes[5] * this->sizes[4] + x4 * this->sizes[5] + x5; }




	__host__ __device__ T operator()(size_t x0) const {  return data[x0]; }
	__host__ __device__ T& operator()(size_t x0) { return data[x0]; }
	__host__ __device__ T operator()(size_t x0, size_t x1) const { static_assert(RANK == 2, ""); return data[x0*this->sizes[1] + x1 ]; }
	__host__ __device__ T& operator()(size_t x0, size_t x1) { static_assert(RANK == 2, ""); return data[x0 *this->sizes[1] + x1 ]; }
	__host__ __device__ T operator()(size_t x0, size_t x1, size_t x2) const { static_assert(RANK == 3, ""); return data[x0*this->sizes[1]*this->sizes[2] + x1 * this->sizes[2] + x2 ]; }
	__host__ __device__ T& operator()(size_t x0, size_t x1, size_t x2) { static_assert(RANK == 3, ""); return data[x0 * this->sizes[1] * this->sizes[2] + x1 * this->sizes[2] + x2]; }
	__host__ __device__ T operator()(size_t x0, size_t x1, size_t x2, size_t x3) const { static_assert(RANK == 4, ""); return data[x0 *this->sizes[3]*this->sizes[2] *this->sizes[1] + x1 *this->sizes[3]*this->sizes[2] + x2 * this->sizes[3] +x3]; }
	__host__ __device__ T& operator()(size_t x0, size_t x1, size_t x2, size_t x3) { static_assert(RANK == 4, ""); return data[x0 * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[3] * this->sizes[2] + x2 * this->sizes[3] + x3];	}
	__host__ __device__ T operator()(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4) const { static_assert(RANK == 5, ""); return data[x0 * this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[4] * this->sizes[3] + x3*this->sizes[4] + x4]; }
	__host__ __device__ T& operator()(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4) {static_assert(RANK == 5, ""); return data[x0 * this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[4] * this->sizes[3] + x3 * this->sizes[4] + x4];}
	__host__ __device__ T operator()(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4, size_t x5) const { static_assert(RANK == 6, ""); return data[x0 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[5]* this->sizes[4] * this->sizes[3] + x3 * this->sizes[5] * this->sizes[4] + x4*this->sizes[5] + x5]; }
	__host__ __device__ T& operator()(size_t x0, size_t x1, size_t x2, size_t x3, size_t x4, size_t x5) { static_assert(RANK == 6, "");  return data[x0 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] * this->sizes[1] + x1 * this->sizes[5] * this->sizes[4] * this->sizes[3] * this->sizes[2] + x2 * this->sizes[5] * this->sizes[4] * this->sizes[3] + x3 * this->sizes[5] * this->sizes[4] + x4 * this->sizes[5] + x5];}


	// template< size_t NEWRANK>
	// Tensor<T,NEWRANK> reshape(std::initializer_list<size_t> newSizes) {
	// 	Tensor<T, NEWRANK> other(this->data, this->size, this->ref_counter, newSizes);
	// 	return other;
	// }

	__host__ __device__ const size_t& getSize() const { return this->size; }
	__host__ __device__ const size_t& getDim(size_t idx) const { return this->sizes[idx]; }
	__host__ __device__ T* getData() const { return this->data; }
	void prefetechToDevice_async() {
		int device = -1;
		CHECK_CUDA_ERROR(cudaGetDevice(&device));
		CHECK_CUDA_ERROR(cudaMemPrefetchAsync(this->data, this->size * sizeof(T), device, NULL));
	}
	void prefetechToHost_async() {
		int device = -1;
		CHECK_CUDA_ERROR(cudaMemPrefetchAsync(this->data, this->size * sizeof(T), device, NULL));
	}

	void prefetechToDevice() {
		int device = -1;
		CHECK_CUDA_ERROR(cudaGetDevice(&device));
		CHECK_CUDA_ERROR(cudaMemPrefetchAsync(this->data, this->size * sizeof(T), device, NULL));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}
	void prefetechToHost() {
		int device = -1;
		CHECK_CUDA_ERROR(cudaMemPrefetchAsync(this->data, this->size * sizeof(T), device, NULL));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}
	Tensor<T, RANK> copy() const  {
		Tensor<T, RANK> other(*this);
		other.data = nullptr;
		other.ref_counter = new size_t(1);
		// std::cout << "alloc" << std::endl;
		CHECK_CUDA_ERROR(cudaMallocManaged((void**)&other.data, other.size * sizeof(T)));
		memcpy(other.data, this->data, sizeof(T) * this->size);
		(*ref_counter)--;
		IF_DEBUG(allocedBytes += this->size * sizeof(T));
		IF_DEBUG(allocedTensors++);
		IF_DEBUG(other._UID = allocedTensors);
		IF_DEBUG(std::cout << "alloc copy: \t[" << this->_UID <<", "<< this->size * sizeof(T)<< " Bytes], alloced: "<< allocedBytes<< std::endl);
		return other;
	}

	void printPoint(size_t i) const{
		std::cout << "Tensor at: \n";
		assert(RANK == 4);
		std::cout << i / (this->getDim(1) *this->getDim(2) * this->getDim(3)) << ", "; 
		i = i % (this->getDim(1) *this->getDim(2) * this->getDim(3));
		std::cout << i / (this->getDim(2) * this->getDim(3)) << ", "; 
		i = i % (this->getDim(2) * this->getDim(3));
		std::cout << i / (this->getDim(3)) << ", "; 
		i = i % (this->getDim(3));
		std::cout << i ; 
		std::cout << std::endl;
	}

	void printDims()const {
		std::cout << "Tensor with dims: ";
		for (size_t i = 0; i < RANK; i++)
		{
			std::cout << this->sizes[i] << ", ";
		}
		std::cout << std::endl;
	}

	void randomize(unsigned int seed, float min, float max) {
		std::mt19937_64 rng(seed);
		std::uniform_real_distribution<float> unif(min, max);
	
		for (int b = 0; b < getSize(); b++)
			data[b] = unif(rng);
	}

	bool sameDims(const Tensor<T, RANK>& other) const {
		if (this->size != other.size) return false;
		for (size_t i = 0; i < RANK; i++)
		{
			if (this->sizes[i] != other.sizes[i]) return false;
		}
		return true;
	}

		/*
	Creates a Tensor of the given size. The Tensor is initialized with the data contained in the 'other' Tensor.
	Data that doesnt fit into the new Tensor is discarded. And the new Tensor is padded with zeros.
	padding: number of zeros to add to each side of the Tensor for coordinates 2 and 3. Coordinates 0 and 1 are not padded.
	*/
	Tensor<T,RANK> copyWithPadding(size_t padding) const{
		assert(RANK == 4);
		assert(padding >= 0);
		std::vector<size_t> newSizes = {this->sizes[0], this->sizes[1], this->sizes[2] + 2 * padding, this->sizes[3] + 2 * padding};
		Tensor<T,RANK> newTensor(newSizes);
		for (size_t i = 0; i < this->sizes[0]; i++)
		{
			for (size_t j = 0; j < this->sizes[1]; j++)
			{
				for (size_t k = 0; k < this->sizes[2]; k++)
				{
					for (size_t l = 0; l < this->sizes[3]; l++)
					{
						newTensor(i,j,k,l) = 0;
					}
				}
			}
		}

		for (size_t i = 0; i < this->sizes[0]; i++)
		{
			for (size_t j = 0; j < this->sizes[1]; j++)
			{
				for (size_t k = 0; k < this->sizes[2]; k++)
				{
					for (size_t l = 0; l < this->sizes[3]; l++)
					{
						newTensor(i,j,k+padding,l+padding) = (*this)(i,j,k,l);
					}
				}
			}
		}
		return newTensor;
	}
};

#endif // TENSOR_H