#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include </home/mallaev/cuda/life/CLI/CLI11.hpp>

typedef uint cell;

namespace cuda_kernels{
	__host__ __device__ int index_from_coordinates(int column, int row, int level, int length, int width, int heighth)
	{
		return ((column + length) % length) + ((row + width) % width) * length + ((level + heighth) % heighth) * length * width;
	}

	__host__ __device__ int calc_neighbours(cell* grid, int column, int row, int level, int length, int width, int heighth)
	{
		return grid[index_from_coordinates(column + -1, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + -1, row + 1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 0, row + 1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + -1, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 0, level + 1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + -1, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + 0, length, width, heighth)] + grid[index_from_coordinates(column + 1, row + 1, level + 1, length, width, heighth)];
	}

	__global__ void calc_next_generation_all_global(cell* current_grid, cell* next_grid, int length, int width, int heighth, int llive, int rlive, int lborn, int rborn)
	{ 
		// Call to global memory approx 28 times per cell - very slow
		int column = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int level = blockIdx.z * blockDim.z + threadIdx.z;

		int place = index_from_coordinates(column, row, level, length, width, heighth);

		if (column < length and row < width and level < heighth)
		{
			cell state = current_grid[place]; // slow
			int neighbours = calc_neighbours(current_grid, column, row, level, length, width, heighth); // 26 * slow

			if ((state == 0 and neighbours >= lborn and neighbours <= rborn) or (state == 1 and neighbours >= llive and neighbours <= rlive))
			{
				next_grid[place] = 1;
			} else {
				next_grid[place] = 0;
			}
		}
	}

	__global__ void calc_next_generation_shared_areas(cell* current_grid, cell* next_grid, int length, int width, int heighth, int llive, int rlive, int lborn, int rborn)
	{
		// Call to global memory approx 2 times per cell - fast
		int column = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
		int row = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
		int level = blockIdx.z * (blockDim.z - 2) + threadIdx.z - 1;

		if (column < length + 1 and row < width + 1 and level < heighth + 1)
		{
			int place = index_from_coordinates(column, row, level, length, width, heighth);

			__shared__ cell area[8192]; // 32kb of shared memory

			area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = current_grid[place]; // slow

			__syncthreads();

			// now we can calculate neighbours fast in a slightly smaller area
			if (threadIdx.x > 0 and threadIdx.x < blockDim.x - 1 and threadIdx.y > 0 and threadIdx.y < blockDim.y - 1 and threadIdx.z > 0 and threadIdx.z < blockDim.z - 1 and column != length and row != width and level != heighth)
			{
				int neighbours;
				neighbours = calc_neighbours(area, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z); // fast

				cell state = area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y];
				
				if ((state == 0 and neighbours >= lborn and neighbours <= rborn) or (state == 1 and neighbours >= llive and neighbours <= rlive))
				{
					next_grid[place] = 1;
				} else {
					next_grid[place] = 0;
				}
			}
		}
	}

	__global__ void calculate_n_generations(int n, cell* current_grid, cell* next_grid, int length, int width, int heighth, int llive, int rlive, int lborn, int rborn)
	{
		//copy areas for each block into it's shared memory and then calculate neighbours
		int column = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
		int row = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
		int level = blockIdx.z * (blockDim.z - 2) + threadIdx.z - 1;

		if (column < length + 1 and row < width + 1 and level < heighth + 1)
		{
			int place = index_from_coordinates(column, row, level, length, width, heighth);

			__shared__ cell area[8192]; // 32kb of shared memory

			if (threadIdx.x > 0 and threadIdx.x < blockDim.x - 1 and threadIdx.y > 0 and threadIdx.y < blockDim.y - 1 and threadIdx.z > 0 and threadIdx.z < blockDim.z - 1 and column != length and row != width and level != heighth)
			{
				area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = current_grid[place]; // pull inner cells
			}

			int neighbours;
			int state;

			cell* now = current_grid;
			cell* next = next_grid;
			cell* temp;

			for (int i = 0; i < n; ++i)
			{
				if (threadIdx.x == 0 or threadIdx.x == blockDim.x - 1 or threadIdx.y == 0 or threadIdx.y == blockDim.y - 1 or threadIdx.z == 0 or threadIdx.z == blockDim.z - 1 or column == length or row == width or level == heighth)
				{
					area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = now[place]; // pull outer cells
				}

				__syncthreads();

				neighbours = calc_neighbours(area, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z); // fast

				state = area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y];
					
				if ((state == 0 and neighbours >= lborn and neighbours <= rborn) or (state == 1 and neighbours >= llive and neighbours <= rlive))
				{
					area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = 1;
				} else {
					area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y] = 0;
				}

				if (threadIdx.x == 1 or threadIdx.x == blockDim.x - 2 or column == length - 1 or threadIdx.y == 1 or threadIdx.y == blockDim.y - 2 or row == width - 1 or threadIdx.z == 1 or threadIdx.z == blockDim.z - 2 or level == heighth - 1)
				{
					next[place] = area[threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y];
				}

				temp = now;
				now = next;
				next = temp;

				__syncthreads();
			}
		}
	}
}

void read_size(std::string filename, int* length, int* width, int* heighth){
	std::ifstream data(filename);
	std::string line;
	std::getline(data, line);

	std::stringstream lineStream(line);
	std::vector<std::string> parsedRow;
	std::string cell;

	while(std::getline(lineStream, cell, ' '))
    {
        parsedRow.push_back(cell);
    }

    *length = std::stoi(parsedRow[0]);
    *width = std::stoi(parsedRow[1]);
    *heighth = std::stoi(parsedRow[2]);
}

void read_input(std::string filename, cell* field_h){
	std::ifstream data(filename);
	std::string line;
	std::getline(data, line);

	std::stringstream lineStream(line);
	std::vector<std::string> parsedRow;
	std::string cell;

	while(std::getline(lineStream, cell, ' '))
    {
        parsedRow.push_back(cell);
    }

    int length = std::stoi(parsedRow[0]);
    int width = std::stoi(parsedRow[1]);
    int heighth = std::stoi(parsedRow[2]);

    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream, cell ,' '))
        {
            parsedRow.push_back(cell);
        }
        if (std::stoi(parsedRow[0]) != 0 or std::stoi(parsedRow[1]) != 0 or std::stoi(parsedRow[2]) != 0 or std::stoi(parsedRow[3]) != 0)
        {
        	field_h[std::stoi(parsedRow[1]) + std::stoi(parsedRow[2]) * length + std::stoi(parsedRow[3]) * length * width] = std::stoi(parsedRow[0]);
        }
    }
}

void append_state_to_file(std::string filename, cell* field_d, int length, int width, int heighth){
	cell field_h[length * width * heighth];
	size_t size = length * width * heighth * sizeof(int);
	cudaMemcpy(field_h, field_d, size, cudaMemcpyDeviceToHost);

	std::ofstream file;
	file.open(filename, std::ios_base::app);

	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			for (int k = 0; k < heighth; ++k)
			{
				int place = i + j * length + k * length * width;
				if (field_h[place] != 0){
					file << field_h[place] << " " << i << " " << j << " " << k << "\n"; 
				}
			}
		}
	}
	file << "0 0 0 0\n";
}

int main(int argc, char** argv){

	CLI::App app{"Cuda game of life"};

	std::string input_filename = "glider.in";
	app.add_option("-i,--input", input_filename, "Input filename");

	std::string output_filename = "test.out";
	app.add_option("-o,--output", output_filename, "Output filename");

	int g = 100;
	app.add_option("-g,--generations", g, "Number of generations to calculate");

	dim3 threads_per_block(8, 8, 8);
	//app.add_option("-b,--block", threads_per_block, "Number of threads per block (3D)");

	std::vector<int> rules = {5, 7, 6, 7};
	app.add_option("-r,--rules", rules, "Number of threads per block (3D)");

	int type = 1;
	app.add_option("-t,--type", type, "0 - all global, 1 - shared areas, 2 - continuous shared");

	int skip = 1;
	app.add_option("-s,--skip", skip, "how often to write state (only for continuous shared)");

	CLI11_PARSE(app, argc, argv);


	int length, width, heighth;

	read_size(input_filename, &length, &width, &heighth);

	cell field_h[length * width * heighth];

	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			for (int k = 0; k < heighth; ++k)
			{
				field_h[i + j * length + k * length * width] = 0;
			}
		}
	}

	read_input(input_filename, field_h);

	cell* field0_d;
	cell* field1_d;
	size_t size = length * width * heighth * sizeof(cell);
	cudaMalloc(&field0_d, size);
	cudaMalloc(&field1_d, size);

	cudaMemcpy(field0_d, field_h, size, cudaMemcpyHostToDevice);

	std::ofstream ofs;
	ofs.open(output_filename, std::ofstream::out | std::ofstream::trunc);
	ofs.close();

	dim3 bpg(length / threads_per_block.x + 1, width / threads_per_block.y + 1, heighth / threads_per_block.z + 1);

	for (int i = 0; i < g; ++i)
	{
		append_state_to_file(output_filename, field0_d, length, width, heighth);
		switch (type)
		{
			case 0:
				cuda_kernels::calc_next_generation_all_global<<<bpg, threads_per_block>>>(field0_d, field1_d, length, width, heighth, rules[0], rules[1], rules[2], rules[3]);
				break;
			case 1:
				cuda_kernels::calc_next_generation_shared_areas<<<bpg, threads_per_block>>>(field0_d, field1_d, length, width, heighth, rules[0], rules[1], rules[2], rules[3]);
				break;
			case 2:
				cuda_kernels::calculate_n_generations<<<bpg, threads_per_block>>>(skip, field0_d, field1_d, length, width, heighth, rules[0], rules[1], rules[2], rules[3]);
				break;
		}
		cudaPeekAtLastError();
        cudaDeviceSynchronize();

		std::swap(field0_d, field1_d);
	}

return 0;
}
