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

#prefix      ?= /usr/local
#exec_prefix ?= $(prefix)
#libdir      ?= $(exec_prefix)/lib
#bindir      ?= $(exec_prefix)/bin
#includedir  ?= $(prefix)/include
#
#DESTDIR := $(abspath $(DESTDIR))
#DESTLIB = $(DESTDIR)$(libdir)
#DESTBIN = $(DESTDIR)$(bindir)
#DESTINC = $(DESTDIR)$(includedir)
#
#LIB_MAJOR_VER ?= $(shell awk '/\#define GDR_API_MAJOR_VERSION/ { print $$3 }' include/gdrapi.h | tr -d '\n')
#LIB_MINOR_VER ?= $(shell awk '/\#define GDR_API_MINOR_VERSION/ { print $$3 }' include/gdrapi.h | tr -d '\n')

PREFIX ?= /usr/local
SRC_INSTALL_DIR ?= $(PREFIX)/src/doca_gpunetio

VERBOSE ?= 0

MAJOR_VER := 0
MINOR_VER := 1
PATCH_VER := 0

VER := $(MAJOR_VER).$(MINOR_VER).$(PATCH_VER)

INC_DIR := include
DOC_DIR := doc
BLD_DIR := build
LIB_DIR := lib
SCRIPT_DIR := scripts
OBJ_DIR := obj
SRC_DIR := src
TOP_DIR := $(CURDIR)

USE_CUDA_WRAPPER ?= 0
USE_NET_WRAPPER ?= 0

CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?= /opt/mpi/openmpi
NVCC ?= $(CUDA_HOME)/bin/nvcc
MPICXX := $(MPI_HOME)/bin/mpicxx
CUDA_ARCH ?= 80

CXXFLAGS ?= -O3
LDFLAGS ?= -L$(CUDA_HOME)/lib64 -lcudart_static

INCLUDES := $(wildcard $(INC_DIR)/*)
SCRIPTS := $(wildcard $(SCRIPT_DIR)/*)
DOXYFILE := $(DOC_DIR)/Doxyfile
MANIFEST := $(OBJ_DIR)/manifest

PACKAGE := $(BLD_DIR)/libdoca_gpunetio-$(VER).tgz
DOC_HTML_PACKAGE := $(BLD_DIR)/libdoca_gpunetio-html-$(VER).tgz
DOC_LATEX_PACKAGE := $(BLD_DIR)/libdoca_gpunetio-latex-$(VER).tgz

LIB_VER := $(VER)
LIB_BASENAME := libdoca_gpunetio_host.so
LIB_SONAME := $(LIB_BASENAME).$(MAJOR_VER)
LIB_FULL_NAME := $(LIB_BASENAME).$(LIB_VER)
LIBS := $(LIB_DIR)/$(LIB_BASENAME) $(LIB_DIR)/$(LIB_BASENAME).$(MAJOR_VER) $(LIB_DIR)/$(LIB_BASENAME).$(VER)

LIBSRCS := doca_verbs_qp.cpp doca_verbs_cq.cpp doca_verbs_device_attr.cpp doca_verbs_umem.cpp doca_verbs_srq.cpp doca_verbs_uar.cpp doca_gpunetio.cpp doca_gpunetio_log.cpp doca_gpunetio_high_level.cpp doca_gpunetio_gdrcopy.cpp
ifeq ($(USE_CUDA_WRAPPER), 1)
LIBSRCS += doca_verbs_cuda_wrapper.cpp
CXXFLAGS += -DDOCA_VERBS_USE_CUDA_WRAPPER
else
LDFLAGS += -lcuda
endif
ifeq ($(USE_NET_WRAPPER), 1)
LIBSRCS += doca_verbs_mlx5dv_wrapper.cpp doca_verbs_ibv_wrapper.cpp
CXXFLAGS += -DDOCA_VERBS_USE_NET_WRAPPER
else
LDFLAGS += -libverbs -lmlx5
endif

LIBOBJS := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(LIBSRCS))

LIB_CXXFLAGS := $(CXXFLAGS) -Wall

EXAMPLES_SUBDIRS := examples/gpunetio_verbs_put_bw
EXAMPLES_SUBDIRS += examples/gpunetio_verbs_write_lat

all: lib examples

version:
	@ echo "$(VER)"

install_src:
	@ echo "Installing source code to $(SRC_INSTALL_DIR)"
	mkdir -p $(SRC_INSTALL_DIR)
	install -m 755 Makefile $(SRC_INSTALL_DIR)/
	install -m 644 README.md $(SRC_INSTALL_DIR)/
	install -d $(SRC_INSTALL_DIR)/include
	install -d $(SRC_INSTALL_DIR)/src
	install -d $(SRC_INSTALL_DIR)/examples
	install -d $(SRC_INSTALL_DIR)/$(INC_DIR)/common
	install -d $(SRC_INSTALL_DIR)/$(INC_DIR)/device
	install -d $(SRC_INSTALL_DIR)/$(INC_DIR)/host
	install -m 644 $(wildcard $(INC_DIR)/common/*.h $(INC_DIR)/common/*.cuh) $(SRC_INSTALL_DIR)/$(INC_DIR)/common/
	install -m 644 $(wildcard $(INC_DIR)/device/*.h $(INC_DIR)/device/*.cuh) $(SRC_INSTALL_DIR)/$(INC_DIR)/device/
	install -m 644 $(wildcard $(INC_DIR)/host/*.h $(INC_DIR)/host/*.cuh) $(SRC_INSTALL_DIR)/$(INC_DIR)/host/
	install -m 644 $(wildcard $(INC_DIR)/*.h $(INC_DIR)/*.cuh) $(SRC_INSTALL_DIR)/$(INC_DIR)/
	install -m 755 $(SCRIPT_DIR)/configure $(SRC_INSTALL_DIR)/$(SCRIPT_DIR)/
	install -m 644 $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.hpp $(SRC_DIR)/*.h $(SRC_DIR)/*.cu) $(SRC_INSTALL_DIR)/src/
	install -d $(SRC_INSTALL_DIR)/$(TEST_SRC_DIR) $(SRC_INSTALL_DIR)/$(PERF_SRC_DIR) $(SRC_INSTALL_DIR)/$(API_SRC_DIR) $(SRC_INSTALL_DIR)/$(TEST_UTIL_DIR)

$(DOC_LATEX_PACKAGE): $(INCLUDES) $(DOXYFILE)
	@ mkdir -p $(BLD_DIR)
	doxygen $(DOXYFILE) > $(BLD_DIR)/doxygen.log 2>&1
	tar -C $(BLD_DIR) -czf $(DOC_LATEX_PACKAGE) latex

$(DOC_HTML_PACKAGE): $(DOC_LATEX_PACKAGE)
	tar -C $(BLD_DIR) -czf $(DOC_HTML_PACKAGE) html

lib: $(OBJ_DIR) $(LIB_DIR)/$(LIB_BASENAME)

manifest: $(OBJ_DIR) $(MANIFEST)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(MANIFEST): $(LIBOBJS)
	@echo "Generating manifest"
	@echo $(abspath $(LIBOBJS)) > $(MANIFEST)

$(LIB_DIR)/$(LIB_BASENAME): $(LIB_DIR)/$(LIB_SONAME)
	ln -sf $(LIB_SONAME) $@

$(LIB_DIR)/$(LIB_SONAME): $(LIB_DIR)/$(LIB_FULL_NAME)
	ln -sf $(LIB_FULL_NAME) $@

$(LIB_DIR)/$(LIB_FULL_NAME): $(LIBOBJS)
	@ echo "Linking library $@"
	@ mkdir -p $(LIB_DIR)
	$(CXX) $(LIB_CXXFLAGS) -shared -Wl,-soname,$(LIB_SONAME) -o $@ $^ $(LDFLAGS)

ifeq ($(and $(USE_CUDA_WRAPPER),$(USE_NET_WRAPPER)),1)
$(LIBOBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
else
$(LIBOBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp lib_configure
endif
	@ echo "Compiling $< -> $@"
	$(CXX) $(LIB_CXXFLAGS) -c $< -o $@ -I$(INC_DIR) -I$(CUDA_HOME)/include -fPIC

lib_configure:
	@ echo "Configuring library"
	$(SCRIPT_DIR)/configure -c $(CUDA_HOME)/bin/nvcc -a sm_$(CUDA_ARCH)

clean:
	@ echo "Cleaning build directories and objects"
	rm -rf $(BLD_DIR) $(LIB_DIR) $(OBJ_DIR) $(TEST_BUILD_DIR)
	for d in $(EXAMPLES_SUBDIRS); do \
		$(MAKE) -C $$d clean; \
	done

examples: $(EXAMPLES_SUBDIRS)
$(EXAMPLES_SUBDIRS): lib_configure
		$(MAKE) -C $@ \
		TOP_DIR=$(TOP_DIR)

.PHONY: clean all lib_configure lib manifest version install_src $(EXAMPLES_SUBDIRS)

ifeq ($(VERBOSE), 0)
.SILENT:
endif
