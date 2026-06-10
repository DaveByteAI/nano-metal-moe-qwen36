CC ?= clang
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC := $(ROOT)src
INCLUDE := $(ROOT)include
METAL := $(ROOT)metal

CFLAGS := -O3 -Wall -Wextra -fobjc-arc -fobjc-exceptions -std=gnu11 -I$(INCLUDE)
FRAMEWORKS := -framework Foundation -framework Metal -framework Accelerate

SOURCES := \
	$(SRC)/main.m \
	$(SRC)/app_config.m \
	$(SRC)/manifest.m \
	$(SRC)/tokenizer.m \
	$(SRC)/math.m \
	$(SRC)/runtime.m \
	$(SRC)/expert_io.m \
	$(SRC)/chat.m \
	$(SRC)/backend/metal_backend.m

all: nmoe

nmoe: $(SOURCES) $(METAL)/kernels.metal
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(FRAMEWORKS)

clean:
	rm -f nmoe

.PHONY: all clean
