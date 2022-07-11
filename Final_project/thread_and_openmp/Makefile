CXX := g++
CXXFLAGS := -I./objs -O3 -std=c++17 -Wall

OBJDIR = objs
OBJS := $(OBJDIR)/constants.o $(OBJDIR)/load_data.o $(OBJDIR)/train.o

default: serial thread omp

.PHONY: dirs clean

dirs:
	/bin/mkdir -p $(OBJDIR)/

clean:
	rm -rf ./objs serial thread

$(OBJDIR)/%.o: %.cpp dirs
		$(CXX) $(CXXFLAGS) -c $< -o $@

serial: serial.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $< -o $@

thread: thread.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

omp: omp.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@
