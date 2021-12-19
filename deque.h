#pragma once

#include <initializer_list>
#include <algorithm>
#include <iostream>

class Chunk {
    int* data_;

    Chunk() : data_(nullptr) {
    }

    ~Chunk() {
        if (data_ != nullptr) {
            delete[] data_;
        }
    }

    void AllocateZero() {
        data_ = new int[128];
        for (int i = 0; i < 128; ++i) {
            data_[i] = 0;
        }
    }

    int operator[](size_t index) const {
        return data_[index];
    }

    int& operator[](size_t index) {
        return data_[index];
    }

    friend class Deque;
};

class Deque {
public:
    Deque() {
        chunks_ = new Chunk[1];
        AllocateChunks();
    }

    ~Deque() {
        delete[] chunks_;
    }

    Deque(const Deque& rhs) {
        number_of_chunks_ = rhs.number_of_chunks_;
        chunks_ = new Chunk[number_of_chunks_];
        AllocateChunks();

        for (int i = 0; i < number_of_chunks_; ++i) {
            for (int j = 0; j < chunk_size_; ++j) {
                chunks_[i][j] = rhs.chunks_[i][j];
            }
        }

        begin_ = rhs.begin_;
        size_ = rhs.size_;
        end_ = rhs.end_;
    }

    Deque(Deque&& rhs) {
        Swap(rhs);
    }

    explicit Deque(size_t size) {
        size_ = size;
        end_ = size_;
        number_of_chunks_ = size_ / chunk_size_ + 1;
        chunks_ = new Chunk[number_of_chunks_];
        AllocateChunks();
    }

    Deque(std::initializer_list<int> list) {
        size_ = list.size();
        end_ = size_;
        number_of_chunks_ = size_ / chunk_size_ + 1;
        chunks_ = new Chunk[number_of_chunks_];
        AllocateChunks();

        for (int i = 0; i < size_; ++i) {
            this->operator[](i) = *(list.begin() + i);
        }
    }

    Deque& operator=(Deque rhs) {
        Swap(rhs);
        return *this;
    }

    void Swap(Deque& rhs) {
        std::swap(chunks_, rhs.chunks_);
        std::swap(number_of_chunks_, rhs.number_of_chunks_);
        std::swap(begin_, rhs.begin_);
        std::swap(size_, rhs.size_);
        std::swap(end_, rhs.end_);
    }

    void PushBack(int value) {
        auto [new_end_chunk_index, new_end_element_index] = Position(end_);
        auto [new_begin_chunk_index, new_begin_element_index] = Position(begin_);

        if (new_end_chunk_index == new_begin_chunk_index &&
            new_end_element_index <= new_begin_element_index && size_ != 0) {
            ReallocateUp(new_begin_chunk_index);
        }

        auto [chunk_index, element_index] = Position(end_);
        ++size_;
        ++end_;

        if (end_ == number_of_chunks_ * chunk_size_) {
            end_ = 0;
        }
        chunks_[chunk_index][element_index] = value;
    }

    void PopBack() {
        --size_;
        --end_;
        if (end_ < 0) {
            end_ = number_of_chunks_ * chunk_size_ - 1;
        }
    }

    void PushFront(int value) {
        auto [new_end_chunk_index, new_end_element_index] = Position(end_);
        auto [new_begin_chunk_index, new_begin_element_index] = Position(begin_ - 1);

        if (new_end_chunk_index == new_begin_chunk_index &&
            new_end_element_index <= new_begin_element_index) {
            new_begin_chunk_index = Position(begin_).first;
            ReallocateUp(new_begin_chunk_index);
        }

        --begin_;
        ++size_;
        if (begin_ < 0) {
            begin_ = number_of_chunks_ * chunk_size_ - 1;
        }

        auto [chunk_index, element_index] = Position(begin_);
        chunks_[chunk_index][element_index] = value;
    }

    void PopFront() {
        ++begin_;
        --size_;
        if (begin_ == number_of_chunks_ * chunk_size_) {
            begin_ = 0;
        }
    }

    int& operator[](size_t ind) {
        auto [chunk_index, element_index] = Position(ind + begin_);
        return chunks_[chunk_index][element_index];
    }

    int operator[](size_t ind) const {
        auto [chunk_index, element_index] = Position(ind + begin_);
        return chunks_[chunk_index][element_index];
    }

    size_t Size() const {
        return size_;
    }

    void Clear() {
        Deque empty;
        Swap(empty);
    }

public:
    const int chunk_size_ = 128;
    Chunk* chunks_ = nullptr;
    int number_of_chunks_ = 1;

public:
    int begin_ = 0;
    int size_ = 0;
    int end_ = 0;

    std::pair<int, int> Position(int index) const {
        if (index == -1) {
            return std::make_pair(number_of_chunks_ - 1, chunk_size_ - 1);
        }
        int chunk_number = index / chunk_size_;
        if (chunk_number >= number_of_chunks_) {
            chunk_number -= number_of_chunks_;
        }
        return std::make_pair(chunk_number, index % chunk_size_);
    }

    void ReallocateUp(int slice) {
        Chunk* new_chunks = new Chunk[number_of_chunks_ * 2];
        for (int i = 0; i < slice; ++i) {
            std::swap(new_chunks[i].data_, chunks_[i].data_);
        }
        for (int i = slice; i < slice + number_of_chunks_; ++i) {
            new_chunks[i].AllocateZero();
        }
        for (int i = slice + number_of_chunks_; i < 2 * number_of_chunks_; ++i) {
            std::swap(new_chunks[i].data_, chunks_[i - number_of_chunks_].data_);
        }

        begin_ += number_of_chunks_ * chunk_size_;
        number_of_chunks_ *= 2;
        end_ = (begin_ + size_) % (number_of_chunks_ * chunk_size_);
        delete[] chunks_;
        chunks_ = new_chunks;
    }

    void AllocateChunks() {
        for (int i = 0; i < number_of_chunks_; ++i) {
            chunks_[i].AllocateZero();
        }
    }
};
