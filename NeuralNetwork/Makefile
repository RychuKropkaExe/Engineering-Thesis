# Project
TESTS_DIR := ./test

FAST_MATRIX_DIR := ./src/FastMatrix
LAYER_DIR := ./src/Layer
MODEL_DIR := ./src/Model
TRAINING_DATA_DIR := ./src/TrainingData

HDRS_DIR := $(FAST_MATRIX_DIR) $(LAYER_DIR) $(MODEL_DIR) $(TRAINING_DATA_DIR)

#KTÓRY 
MODE := app #Albo app albo test

#PROJECT FILES
SRC := $(wildcard $(FAST_MATRIX_DIR)/*.cpp) $(wildcard $(LAYER_DIR)/*.cpp) $(wildcard $(MODEL_DIR)/*.cpp) $(wildcard $(TRAINING_DATA_DIR)/*.cpp)

HDRS := $(wildcard $(FAST_MATRIX_DIR)/*.h) $(wildcard $(LAYER_DIR)/*.h) $(wildcard $(MODEL_DIR)/*.h) $(wildcard $(TRAINING_DATA_DIR)/*.h)

SRC := $(SRC) $(wildcard $(TESTS_DIR)/*.cpp)
HDRS := $(HDRS) $(wildcard $(TESTS_DIR)/*.h)
EXEC := $(TESTS_DIR)/tests.exe

#COMPILATION
OBJ := $(SRC:%.cpp=%.obj)
DEPS := $(OBJ:%.obj=%.d)

CC := g++
C_FLAGS := -Wall -Wextra -Werror -O3
DEP_FLAGS := -MMD -MP

LIB := pthread
INC := $(foreach h, $(HDRS_DIR), -I$h)
L_INC := $(foreach l, $(LIB), -l$l)


ifeq ($(CC), clang)
	C_FLAGS += -Weverything -Wno-vla -Wno-float-equal -Wno-disabled-macro-expansion
endif

GFLAG := no

ifeq ($(GFLAG), yes)
	GGDB := -ggdb3 -g
endif
C_FLAGS += $(GGDB)

all: $(EXEC)

$(EXEC): $(OBJ)
	@echo "[CREATING EXEC]"
	$(CC) $(C_FLAGS) $(INC) $(OBJ) -o $@ $(L_INC)

%.obj:%.cpp %.d
	@echo "[COMPILATION]"
	$(CC) $(C_FLAGS) $(INC) -c $< -o $@

#CLEAN
clean:
	@echo "[CLEANING EXEC]"
	rm -rf $(EXEC)
	@echo "[CLEANING OBJECTS]"
	rm -rf $(OBJ)
	@echo "[CLEANING DEPENDECIES]"
	rm -rf $(DEPS)

#DEPS 
$(DEPS):

include $(wildcard $(DEPS))