/*
	This file is part of the NRRPGpu program.

	NRRPGpu is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	NRRPGpu is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License

*/

#ifndef NRRP_H_
#define NRRP_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constraints.cuh"
#include "utility.cuh"
#include "io.cuh"
#include "roster.cuh"
#include "device.cuh"
#include "settings.cuh"

#define INTMAX 4294967295
#define WATCHDOG
#define INVERSE_SEARCH
#define LOCAL_SEARCH
#define LOCAL_SEARCH_RWS_TRESHOLD 2000000
#define LOCAL_SEARCH_RSP_TRESHOLD 1000
#define HOMOGENOUS

enum Shift {FREE = 0, NIGHT = 1, AM = 2, PM = 3};

extern unsigned int* _originalRoster; // Original roster array
extern unsigned int _nurseCount; // Number of nurses  for rerostering
extern unsigned int _dayCount; // Number of days for rerostering
extern unsigned int _shiftToReroster; // Number of shifts for rerostering
extern unsigned int _runCount; // Number of runs
extern unsigned int _shiftChangeMaximum; // Maximal number of changes
extern unsigned int _initialChangeCount; // Number of rerostered shifts
extern unsigned int _paralelInstanceCount; // Number of parallel proceeded threads
extern unsigned int _absenceCount; // Number of absences
extern unsigned int _bestUtility; // Best utility function
extern unsigned int* _bestRoster; // List of changes of best roster
extern unsigned int* _shiftList; // Ordered list of shift to reroster
extern unsigned int* _shiftMap; // Bit array for mapping already rostered shifts
extern unsigned int* _shiftMapInit; // Initialization array for bit array for mapping already rostered shifts
extern unsigned int* _freePreferenceMap; // Bit array of shifts in shiftMap that are FREE
extern unsigned int* _absenceArray; // Bit array of absences
extern unsigned int* _currentRoster; // Array for list of changes
extern unsigned int* _currentRosterInit; // Initial array for actual rosters
extern unsigned int _nonFreeDays; // Number of days, when any nurse has some shift
extern unsigned int _fairnessAverage; // Number of duties averaged over the nurses
extern unsigned int* _nursePreferences; // Array of nurse preferences
extern unsigned int* _sdscFreeDays; // Number of demanded free days in seven day sequence
extern unsigned int* _erRoster; // Extended roster - one day for every nurse
extern unsigned int* _erShiftMap; // Extended shift map
extern unsigned int* _staffGapsInitOriginal; // Initial gabs in original roster
extern unsigned int* _staffGapsInit;  // Initial gabs in original roster substracted by absences
extern unsigned int* _staffGaps; // Actual gabs in current roster
extern unsigned int _TESTTotalfeasible;
extern unsigned int _TESTTotalruns;
extern unsigned int _TESTSolved;
extern unsigned int _TESTLocalFound;
extern unsigned int _TESTTotalSolvedRun;
extern float _TESTTotalfirst;
extern float _TESTTotalsecond;
extern float _TESTTotalQuality;
extern float _BPartTime;
#ifdef WATCHDOG
	extern struct timeb _startTime; // The time, when algorithm started
	//extern double _stopTime[]; // Difference between start and stop time
	extern double _stopTime;
	extern unsigned int _inputFileNumb;
#endif

#ifdef LOCAL_SEARCH
	extern unsigned int* _bestShiftList;
#endif

int main(int argc, char *argv[]);
void kerlen_run(unsigned int* shiftList,unsigned int* shiftMap, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* shiftGabs);
void nurseSaveBest(unsigned int* currentRoster, unsigned int* shiftList, int nurse);
void nurseAssign(unsigned int* actualShifts, unsigned int* penalityFunction, unsigned int* currentRoster, unsigned int* nonAssignedShifts, unsigned int* shiftMap, unsigned int* staffGaps, int nurse);
void nurseCompatibilityPenalization(int size, unsigned int* actualShifts, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* shiftMap, unsigned int isFirst, int nurse);
void nurseLoadSecondPhase(unsigned int shiftSecondPhase, unsigned int* actualShifts, int nurse);
void nurseLoadFirstPhase(int index, unsigned int* shiftList, unsigned int* actualShifts, int nurse);
void nurseInitialize(unsigned int* currentRoster, unsigned int* shiftMap, unsigned int* shiftList, unsigned int* nonAssignedShift, unsigned int* staffGaps, int isLocal, float probability, int nurse);
//void run(unsigned int* shiftList,unsigned int* shiftMap, unsigned int* _currentRoster);
void cleaner();
void initialization();
#endif
