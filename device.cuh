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

#ifndef DEVICE_H_
#define DEVICE_H_
#define DEVICE

#include "settings.cuh"

extern unsigned int *dev_shiftMap, *dev_currentRoster, *dev_index, *dev_origRoster, *dev_isFirst, *dev_freePreferenceMap, *dev_nursePreferences, *dev_sdscFreeDays, *dev_erRoster, *dev_erShiftMap;
extern unsigned int *dev_result;

void initializeDevice();
void freeDevice();
__device__ __inline unsigned int dev_getBitOfBitArray(unsigned int* array, int position, unsigned int size);
__device__ __inline void dev_setBitOfBitArray(unsigned int* array, int position, int value, unsigned int dayCount);
__device__ __inline unsigned int dev_getRow(unsigned int index, unsigned int size);
__device__ __inline unsigned int dev_getColumn(unsigned int index, unsigned int size);
__device__ __inline int dev_getOrigRosterShift(unsigned int index, unsigned int* originalRoster, unsigned int size);
__device__ __inline int dev_getRosterShift(unsigned int index, unsigned int* currentRoster, unsigned int size);
__device__ __inline int dev_getExtendedRosterShift(unsigned int index, unsigned int* currentRoster);
__device__ __inline void dev_setRosterShift(unsigned int* roster, unsigned int index, int shift, unsigned int dayCount);
__device__ __inline unsigned int dev_getNursePreference(unsigned int index, unsigned int shift, unsigned int* nursePreferences);
__device__ __inline int dev_isLeftHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster, unsigned int* erRoster, Settings settings);
__device__ __inline int dev_isRightHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster);
__device__ __inline int dev_isHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst, unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings);
__device__ __inline unsigned int dev_CompatibilityPenalization(unsigned int index, int size, unsigned int nurseCount, unsigned int shiftType, unsigned int* originalRoster, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings);

//Heterogenous
__global__ void kernel(unsigned int* index, int size, unsigned int nurseCount, unsigned int* origRoster, unsigned int* shiftMap, unsigned int* isFirst, unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings, unsigned int* result);
unsigned int deviceCompatibilityPenalizationHost(unsigned int* index, int size, unsigned int* shiftMap, unsigned int* isFirst,unsigned int* currentRoster, unsigned int* penaltyFunction);

//Homogenous
__global__ void homogenousKernel(unsigned int* shiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestUtility, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int* bestLock, unsigned int* absenceArray, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, unsigned int* localSearch, unsigned int* runsWithoutSucess, unsigned int* runsWithSameProbability, float* probability, unsigned int* localSearchLock, unsigned int* exit, unsigned int* run, unsigned int runCount, Settings settings, unsigned int nonFreeDays, unsigned int initialChangeCount, unsigned int nurseCount, unsigned int dayCount, unsigned int fairnessAverage, curandState* state, int seed);
__device__ __inline void dev_KernelRun(unsigned int* shiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestUtility, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int* bestLock, unsigned int* absenceArray, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, unsigned int* localSearch, unsigned int* runsWithoutSucess, unsigned int* runsWithSameProbability, unsigned int* localSearchLock, float* probability, unsigned int* exit, unsigned int* run, unsigned int runCount, Settings settings, unsigned int nonFreeDays, unsigned int initialChangeCount, unsigned int nurseCount, unsigned int dayCount, unsigned int fairnessAverage, unsigned int nurse, unsigned int instanceInBlockIndex, curandState* state);
__device__ void lock(unsigned int* mutex);
__device__ void unlock(unsigned int* mutex);
__device__ __inline void dev_nurseSaveBest(unsigned int* currentRoster, unsigned int* shiftList, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int dayCount, unsigned int nurse);
__device__ __inline void dev_nurseAssign(unsigned int* actualShifts, unsigned int* penalityFunction, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* nonAssignedShifts, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* absenceArray, unsigned int* staffGaps, unsigned int dayCount, unsigned int nurse);
__device__ __inline void dev_nurseCompatibilityPenalization(unsigned int nurseCount, int size, unsigned int* actualShifts, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* shiftMap, unsigned int isFirst, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings, unsigned int dayCount, unsigned int nurse);
__device__ __inline void dev_nurseLoadSecondPhase(unsigned int shiftSecondPhase, unsigned int* actualShifts, unsigned int nurse);
__device__ __inline void dev_nurseLoadFirstPhase(int index, unsigned int* shiftList, unsigned int* actualShifts, unsigned int dayCount, unsigned int nurse);
__device__ __inline void dev_nurseInitialize(unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestShiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* shiftList, unsigned int* nonAssignedShift, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int isLocal, unsigned int nonFreeDays, float probability, unsigned int nurseCount, unsigned int dayCount, unsigned int nurse, unsigned int keepShiftList, curandState* state);
__device__ __inline int dev_minShiftTypeConsecutiveConstrainTest(int actual, int NIGHTs, int AMs, int PMs, Settings settings);
__device__ unsigned int dev_nurseUtilityFunction(unsigned int* currentRoster, unsigned int* originalRoster, unsigned int* nursePreference, unsigned int dayCount, Settings settings, unsigned int fairnessAverage, unsigned int nurse);
__device__ unsigned int dev_nurseUtilityFunctionMoz(unsigned int* currentRoster, unsigned int* originalRoster, unsigned int* nursePreference, unsigned int dayCount, Settings settings, unsigned int fairnessAverage, unsigned int nurse);
__device__ unsigned int dev_CompatibilityPenalizationFlat(unsigned int index, int size, unsigned int nurseCount, unsigned int shiftType, unsigned int* originalRoster, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings);

void homogenousKernelCall();

#endif /* ROASTER_H_ */
