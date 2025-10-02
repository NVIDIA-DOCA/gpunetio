# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# === Project settings ===
TARGET    := gpunetio_verbs_write_lat
BUILD_DIR := build

INCLUDE_DIR = -I$(TOP_DIR)/include -I$(TOP_DIR)/include/host -I$(TOP_DIR)/include/common -I../ -I$(TOP_DIR)/src -I$(CUDA_HOME)/include 

# CUDA 13 requires C++17
CXXFLAGS = $(INCLUDE_DIR) -std=c++17 -fPIC # -g3
NVCCFLAGS = $(INCLUDE_DIR) -std=c++17 -arch=sm_$(CUDA_ARCH) -I$(TOP_DIR)/include/device #-G

LDFLAGS = -L$(TOP_DIR)/lib -ldoca_gpunetio_host -L$(CUDA_HOME)/lib64 -libverbs -lmlx5 -lcuda -lcudart

SRCS := $(TOP_DIR)/examples/verbs_common.cpp gpunetio_verbs_write_lat_sample.cpp gpunetio_verbs_write_lat_main.cpp gpunetio_verbs_write_lat_kernel.cu

OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(filter %.cu,$(SRCS))) \
        $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(SRCS)))

# === Compiler setup ===
CXX = g++
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 80

.PHONY: all clean

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CUDA_HOME)/bin/nvcc $(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

# Compile C sources
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CUDA_HOME)/bin/nvcc $(NVCCFLAGS) -c $< -o $@

# Compile C++ sources
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Cleanup
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
