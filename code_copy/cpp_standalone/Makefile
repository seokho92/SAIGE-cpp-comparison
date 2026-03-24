# saige-build mamba env — has all required libs (armadillo, superlu, tbb, yaml-cpp, R, RcppParallel)
SAIGE_BUILD  := /data/home/seokhojeong/.local/share/mamba/envs/saige-build
CXX          := $(SAIGE_BUILD)/bin/x86_64-conda-linux-gnu-c++
CXXFLAGS     ?= -std=c++17 -O2 -DARMA_64BIT_WORD=1 -DRCPP_PARALLEL_USE_TBB=0 -MMD -MP -pthread

# Also used for fallback include/lib probing
CONDA_BASE   ?= /data/home/seokhojeong/anaconda3
# Active conda env (may differ; saige-build takes priority)
CONDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)

RSCRIPT    := $(shell command -v Rscript 2>/dev/null)
PKG_CONFIG := $(shell command -v pkg-config 2>/dev/null)

# -------------------------
# Optional R / Rcpp includes and libs
# -------------------------
# Use R from saige-build env directly (avoids active-env CXX override issues)
SAIGE_RSCRIPT   := $(SAIGE_BUILD)/bin/Rscript
SAIGE_RLIB      := $(SAIGE_BUILD)/lib/R/library
R_HOME          := $(SAIGE_BUILD)/lib/R
RCPP_INC        := $(SAIGE_RLIB)/Rcpp/include
RCPPARM_INC     := $(SAIGE_RLIB)/RcppArmadillo/include
RCPPARAL_INC    := $(SAIGE_RLIB)/RcppParallel/include
RCPPEIG_INC     :=
RCPPARAL_LIBDIR := $(SAIGE_RLIB)/RcppParallel/libs
RCPPARAL_SO     := $(shell ls "$(RCPPARAL_LIBDIR)/RcppParallel.so" 2>/dev/null)

# -------------------------
# pkg-config (Armadillo / yaml-cpp)
# -------------------------
ifeq ($(strip $(PKG_CONFIG)),)
  ARMA_CFLAGS :=
  ARMA_LIBS   :=
  YAML_CFLAGS :=
  YAML_LIBS   :=
else
  ARMA_CFLAGS := $(shell "$(PKG_CONFIG)" --cflags armadillo 2>/dev/null)
  ARMA_LIBS   := $(shell "$(PKG_CONFIG)" --libs armadillo 2>/dev/null)
  YAML_CFLAGS := $(shell "$(PKG_CONFIG)" --cflags yaml-cpp 2>/dev/null)
  YAML_LIBS   := $(shell "$(PKG_CONFIG)" --libs yaml-cpp 2>/dev/null)
endif

# -------------------------
# Include paths
# -------------------------
# Local directory first (picks up cxxopts.hpp bundled alongside sources)
INCLUDES := -I. -I/usr/local/include

# saige-build env (all required headers including yaml-cpp, armadillo, eigen3, RcppParallel)
INCLUDES += -isystem $(SAIGE_BUILD)/include
INCLUDES += -I$(SAIGE_BUILD)/include
INCLUDES += -I$(SAIGE_BUILD)/include/eigen3

# System fallback
INCLUDES += -I/usr/include/eigen3

# Optional R headers
ifneq ($(strip $(R_HOME)),)
  INCLUDES += -I"$(R_HOME)/include"
endif
ifneq ($(strip $(RCPP_INC)),)
  INCLUDES += -I"$(RCPP_INC)"
endif
ifneq ($(strip $(RCPPARM_INC)),)
  INCLUDES += -I"$(RCPPARM_INC)"
endif
ifneq ($(strip $(RCPPARAL_INC)),)
  INCLUDES += -I"$(RCPPARAL_INC)"
endif
ifneq ($(strip $(RCPPEIG_INC)),)
  INCLUDES += -I"$(RCPPEIG_INC)"
endif

# pkg-config cflags
ifneq ($(strip $(YAML_CFLAGS)),)
  INCLUDES += $(YAML_CFLAGS)
endif
ifneq ($(strip $(ARMA_CFLAGS)),)
  INCLUDES += $(ARMA_CFLAGS)
endif

# -------------------------
# Libraries / linker flags
# -------------------------
LIBS := -pthread

# saige-build env libs (has everything: yaml-cpp, armadillo, superlu, tbb, R, RcppParallel)
LIBS += -L$(SAIGE_BUILD)/lib
LIBS += -Wl,-rpath,$(SAIGE_BUILD)/lib

# Core libs
LIBS += -lyaml-cpp -larmadillo -lopenblas -llapack -lsuperlu -ltbb

# pkg-config libs (can duplicate; usually harmless)
ifneq ($(strip $(YAML_LIBS)),)
  LIBS += $(YAML_LIBS)
endif
ifneq ($(strip $(ARMA_LIBS)),)
  LIBS += $(ARMA_LIBS)
endif

# Optional R library
ifneq ($(strip $(R_HOME)),)
  LIBS += -L"$(R_HOME)/lib" -lR
  LIBS += -Wl,-rpath,"$(R_HOME)/lib"
endif

# Optional RcppParallel runtime (link actual .so directly for reliability)
ifneq ($(strip $(RCPPARAL_LIBDIR)),)
  LIBS += -Wl,-rpath,"$(RCPPARAL_LIBDIR)"
endif
ifneq ($(strip $(RCPPARAL_SO)),)
  LIBS += "$(RCPPARAL_SO)"
endif

# -------------------------
# Sources / targets
# -------------------------
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
DEPS    := $(OBJECTS:.o=.d)

TARGET  ?= saige-null

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Profile build: debug symbols + light optimization for meaningful callgrind output
profile: CXXFLAGS := $(filter-out -O2,$(CXXFLAGS)) -O1 -g
profile: clean $(TARGET)

# Run under callgrind (requires CONFIG= to be set, e.g. make callgrind CONFIG=config_test.yaml)
CONFIG ?= config_test.yaml
callgrind: profile
	valgrind --tool=callgrind \
	         --callgrind-out-file=callgrind.out.%p \
	         --collect-jumps=yes \
	         --cache-sim=yes \
	         --branch-sim=yes \
	         ./$(TARGET) -c $(CONFIG)

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)

.PHONY: all clean profile callgrind

-include $(DEPS)