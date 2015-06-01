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

#include "nrrp.cuh"
#include "cuPrintf.cu"

unsigned int *dev_shiftMap, *dev_currentRoster, *dev_index, *dev_origRoster, *dev_isFirst, *dev_freePreferenceMap;
unsigned int *dev_nursePreferences, *dev_sdscFreeDays, *dev_erRoster, *dev_erShiftMap;
unsigned int *dev_shiftList, *dev_shiftMapInit, *dev_currentRosterInit, *dev_bestUtility, *dev_bestRoster;
unsigned int *dev_bestShiftList, *dev_bestLock, *dev_absenceArray, *dev_penalityFunctions;
unsigned int *dev_nonAssignedShifts, *dev_staffGaps, *dev_staffGapsInit, *dev_localSearch, *dev_runsWithoutSucess;
unsigned int *dev_runsWithSameProbability,  *dev_localSearchLock, *dev_exit, *dev_run;
curandState* dev_states;
unsigned int* dev_result;
float* dev_probability;

void initializeDevice()
{
	cudaMalloc( (void**)&dev_shiftMap, sizeof(int)*_nurseCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_currentRoster, sizeof(int)*2*_nurseCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_index, sizeof(int)*_nurseCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_origRoster, sizeof(int)*_nurseCount*2);
	cudaMalloc( (void**)&dev_isFirst, _paralelInstanceCount*sizeof(int));
	cudaMalloc( (void**)&dev_freePreferenceMap, sizeof(int)*_nurseCount);
	cudaMalloc( (void**)&dev_nursePreferences, sizeof(int)*_nurseCount*_dayCount*4);
	cudaMalloc( (void**)&dev_sdscFreeDays, sizeof(unsigned int)*_nurseCount);
	cudaMalloc( (void**)&dev_erRoster, sizeof(unsigned int)*2);
	cudaMalloc( (void**)&dev_erShiftMap, sizeof(unsigned int)*_nurseCount);
	cudaMalloc( (void**)&dev_result, sizeof(int)*_nurseCount*_paralelInstanceCount);
#ifdef HOMOGENOUS
	cudaMalloc( (void**)&dev_shiftList, sizeof(int)*_nurseCount*_dayCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_shiftMapInit, sizeof(int)*_nurseCount);
	cudaMalloc( (void**)&dev_currentRosterInit, sizeof(int)*2*_nurseCount);
	cudaMalloc( (void**)&dev_bestUtility, sizeof(int));
	cudaMalloc( (void**)&dev_bestRoster, sizeof(int)*2*_nurseCount);
	cudaMalloc( (void**)&dev_bestShiftList, sizeof(int)*_nurseCount*_dayCount);
	cudaMalloc( (void**)&dev_bestLock, sizeof(int));
	cudaMalloc( (void**)&dev_absenceArray, sizeof(int)*_nurseCount);
	cudaMalloc( (void**)&dev_penalityFunctions, sizeof(int)*_nurseCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_nonAssignedShifts, sizeof(int)*_nurseCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_staffGaps, sizeof(int)*3*_dayCount*_paralelInstanceCount);
	cudaMalloc( (void**)&dev_staffGapsInit, sizeof(int)*3*_dayCount);
	cudaMalloc( (void**)&dev_localSearch, sizeof(int));
	cudaMalloc( (void**)&dev_runsWithoutSucess, sizeof(int));
	cudaMalloc( (void**)&dev_runsWithSameProbability, sizeof(int));
	cudaMalloc( (void**)&dev_localSearchLock, sizeof(int));
	cudaMalloc( (void**)&dev_exit, sizeof(int));
	cudaMalloc( (void**)&dev_run, sizeof(int));
	cudaMalloc( (void**)&dev_probability, sizeof(float));
	cudaMalloc( (void**)&dev_states, _nurseCount * _paralelInstanceCount * sizeof(curandState));
#endif
	cudaMemcpy( dev_origRoster, _originalRoster, sizeof(int)*_nurseCount*2, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_nursePreferences, _nursePreferences, sizeof(int)*_nurseCount*_dayCount*4, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_sdscFreeDays, _sdscFreeDays, sizeof(unsigned int)*_nurseCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_erRoster, _erRoster, sizeof(unsigned int)*2, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_erShiftMap, _erShiftMap, sizeof(unsigned int)*_nurseCount, cudaMemcpyHostToDevice);
}

void freeDevice()
{
	cudaFree((void *) dev_shiftMap);
	cudaFree((void *) dev_currentRoster);
	cudaFree((void *) dev_index);
	cudaFree((void *) dev_origRoster);
	cudaFree((void *) dev_isFirst);
	cudaFree((void *) dev_freePreferenceMap);
	cudaFree((void *) dev_nursePreferences);
	cudaFree((void *) dev_sdscFreeDays);
	cudaFree((void *) dev_erRoster);
	cudaFree((void *) dev_erShiftMap);
	cudaFree((void *) dev_result);
#ifdef HOMOGENOUS
	cudaFree((void *)dev_shiftList);
	cudaFree((void *)dev_shiftMapInit);
	cudaFree((void *)dev_currentRosterInit);
	cudaFree((void *)dev_bestUtility);
	cudaFree((void *)dev_bestRoster);
	cudaFree((void *)dev_bestShiftList);
	cudaFree((void *)dev_bestLock);
	cudaFree((void *)dev_absenceArray);
	cudaFree((void *)dev_penalityFunctions);
	cudaFree((void *)dev_nonAssignedShifts);
	cudaFree((void *)dev_staffGaps);
	cudaFree((void *)dev_staffGapsInit);
	cudaFree((void *)dev_localSearch);
	cudaFree((void *)dev_runsWithoutSucess);
	cudaFree((void *)dev_runsWithSameProbability);
	cudaFree((void *)dev_localSearchLock);
	cudaFree((void *)dev_exit);
	cudaFree((void *)dev_run);
	cudaFree((void *)dev_probability);
	cudaFree((void *)dev_states);
#endif
}

// Get selected bit from array of bits
__device__ __inline unsigned int dev_getBitOfBitArray(unsigned int* array, int position, unsigned int size)
{
	return array[position/size]&(1<<(position%size));
}

// Set selected bit in array of bits
__device__ __inline void dev_setBitOfBitArray(unsigned int* array, int position, int value, unsigned int dayCount)
{
	int pos = (position/dayCount)*32+(position%dayCount);
	if(value)
		array[pos/32] = array[pos/32]|(1<<(pos%32));
	else
		array[pos/32] = array[pos/32]&(INTMAX - (1<<(pos%32)));
}

// Conversation 1D index to row
__device__ __inline unsigned int dev_getRow(unsigned int index, unsigned int size)
{
	return index/size;
}

// Conversation 1D index to column
__device__ __inline unsigned int dev_getColumn(unsigned int index, unsigned int size)
{
	return index%size;
}

__device__ __inline int dev_getOrigRosterShift(unsigned int index, unsigned int* originalRoster, unsigned int size)
{
	int ind = (index/size)*32+(index%size);
	int indm16 = ind%16;
	return (originalRoster[ind/16]&(3<<(2*(indm16))))>>(2*(indm16));
}

// Return shift at index in current roster
__device__ __inline int dev_getRosterShift(unsigned int index, unsigned int* currentRoster, unsigned int size)
{
	int ind = (index/size)*32+(index%size);
	int indm16 = ind%16;
	return (currentRoster[ind/16]&(3<<(2*(indm16))))>>(2*(indm16));
}

// Return shift at index in extended roster
__device__ __inline int dev_getExtendedRosterShift(unsigned int index, unsigned int* currentRoster)
{
	int ind = index/16;
	int indm16 = index%16;
	return (currentRoster[ind]&(3<<(2*(indm16))))>>(2*(indm16));
}

__device__ __inline void dev_setRosterShift(unsigned int* roster, unsigned int index, int shift, unsigned int dayCount)
{
	int ind = (index/dayCount)*32+(index%dayCount);
	int indm16 = ind%16;
	ind = ind/16;
	roster[ind] = roster[ind]&((((INTMAX << 2)+shift+1) << (2*(indm16)))-1);
	roster[ind] = roster[ind]|(shift << (2*(indm16)));;
}

__device__ __inline unsigned int dev_getNursePreference(unsigned int index, unsigned int shift, unsigned int* nursePreferences)
{
	return nursePreferences[index*4+shift];
}

__device__ __inline int dev_isLeftHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster, unsigned int* erRoster, Settings settings)
{
	int left;
	if(dev_getColumn(index, size) == 0)
	{
		if(settings.isExtendedRoster)
			left = dev_getExtendedRosterShift(dev_getRow(index, size), erRoster);
		else
			return 0;
	}
	else
	{
		left = dev_getRosterShift(index-1, currentRoster, size);
	}
	if(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT)))
		return 1;
	return 0;
}

// Method tests if there is hard constrain broken between actual and next shift
__device__ __inline int dev_isRightHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster)
{
	int right;
	if(dev_getColumn(index, size) == size-1)
		return 0;

	right = dev_getRosterShift(index+1, currentRoster, size);
	if((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))
		return 1;
	return 0;
}

// Method tests if there is hard constrain broken for actual shift
__device__ __inline int dev_isHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst, unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings)
{
	if(isFirst || !settings.isLeftRightConstrainSoft)
	{
		if(dev_isLeftHardConstrainBroken(index, size, shiftType, currentRoster, erRoster, settings)
				|| dev_isRightHardConstrainBroken(index, size, shiftType, currentRoster))
			return 1;
	}
	
	if(settings.isSevenDaySequenceConstrain)
	{
		unsigned int shifts, shift, frees, nurse, shiftMapNurse;
		int tmp = 0;
		int i, startDay, day;
		day = dev_getColumn(index, size);
		nurse = dev_getRow(index, size);
		shiftMapNurse = shiftMap[nurse];
		shiftMapNurse = shiftMapNurse&(~freePreferenceMap[nurse]);
		shifts = 0;
		frees = 0;
		if(settings.isExtendedRoster)
			startDay = day-6;	
		else
			startDay = day>5?day-6:0;
		for(i = startDay; i < startDay+7; i++)
		{
			if(i >= size)
			{
				frees++;
				shifts = shifts<<1;
			}
			else
			{
				if(i != day)
				{
					if(i < 0)
					{
						shift = (dev_getBitOfBitArray(erShiftMap, nurse*size-i, size))?1:0;
					}
					else
					{
						shift = (dev_getBitOfBitArray(&shiftMapNurse, i, size))?1:0;
					}
				}
				else
				{
					shift = (shiftType==FREE)?0:1;
				}
				shifts += shift;
				shifts = shifts << 1;
				if(!shift)
				{
					frees++;
				}
			}
		}
		if(frees < sdscFreeDays[nurse])
		{
			return 1;
		}
		for(i = startDay+7; i<(int)day+7; i++) // Check all other seven-day sequencies
		{
			if(i < (int)size)
				shift = (dev_getBitOfBitArray(&shiftMapNurse, i, size))?1:0;
			else
				return 0;
			if(!shift) frees++;
			if(!((shifts>>(7-tmp))&1))
				frees--;
			if(frees < sdscFreeDays[nurse])
				return 1;
			tmp++;
		}
	}
	/*
	if(settings.isMinMaxHardConstrain)
	{
		unsigned int nurse, countWorking, countConsecutive, countAM, countNIGHT, countNight, countShiftTypeCons, i;
		int actualShift, lastShift;
		countConsecutive = countAM = countNIGHT = countNight = countShiftTypeCons = countWorking = 0;
		lastShift = FREE;
		nurse = dev_getRow(index, size);
		for(i = nurse*size; i < (nurse*size+size); i++)
		{
			if(i == index)
				actualShift = shiftType;
			else
				actualShift = dev_getRosterShift(i, currentRoster, size);
			switch(actualShift)
			{
			case FREE:
				countConsecutive = 0;
				countShiftTypeCons = 0;
				lastShift = FREE;
				break;
			case NIGHT:
				countWorking++;
				countConsecutive++;
				countNIGHT++;
				if((lastShift==FREE)||(lastShift==NIGHT))
					countShiftTypeCons++;
				else
					countShiftTypeCons = 0;
				break;
			case AM:
				countWorking++;
				countConsecutive++;
				countAM++;
				if((lastShift==FREE)||(lastShift==AM))
					countShiftTypeCons++;
				else
					countShiftTypeCons = 0;
				break;
			case PM:
				countWorking++;
				countConsecutive++;
				countNight++;
				if((lastShift==FREE)||(lastShift==PM))
					countShiftTypeCons++;
				else
					countShiftTypeCons = 0;
				break;
			}
			if((countNIGHT > settings.mmhcMaxShiftType)||(countAM > settings.mmhcMaxShiftType)||(countNight > settings.mmhcMaxShiftType)\
				||(countWorking > settings.mmhcMaxWorking)||(countConsecutive > settings.mmhcMaxConsecutive))
				return 1;
		}
	}*/
	return 0;
}

__device__ __inline unsigned int dev_CompatibilityPenalization(unsigned int index, int size, unsigned int nurseCount, unsigned int shiftType, unsigned int* originalRoster, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings)
{
	unsigned int left, right, conCount = 0, result = 0;
	unsigned int i = 2;

	if(dev_getBitOfBitArray(shiftMap, index, size))
	{
		return INTMAX;
	}

	if(dev_isHardConstrainBroken(index, size, shiftType, shiftMap, isFirst, currentRoster, freePreferenceMap, sdscFreeDays, erRoster, erShiftMap, settings))
		return INTMAX;

	if(isFirst)
		return 0;
	if(settings.isMPHeuristic == 1)
	{
		result += 2*settings.mphPenalization;
		if(dev_getColumn(index, size) == 0)
		{
			if(settings.isExtendedRoster)
				left = dev_getExtendedRosterShift(dev_getRow(index, size), erRoster);
			else
				left = FREE;
		}
		else
			left = dev_getRosterShift(index-1, currentRoster, size);

		if(dev_getColumn(index, size) == size-1)
			right = FREE;
		else
			right = dev_getRosterShift(index+1, currentRoster, size);
		if((left!=FREE)||(right!=FREE))
			result -= settings.mphPenalization;
		if((left!=FREE)&&(right!=FREE))
			result -= settings.mphPenalization;
	}

	if(settings.isMPHeuristic == 2)
	{
		result += 2*settings.mphPenalization;
		if(dev_getColumn(index, size) == 0)
		{
			if(settings.isExtendedRoster)
				left = dev_getExtendedRosterShift(dev_getRow(index, size), erRoster);
			else
				left = FREE;
		}
		else
			left = dev_getOrigRosterShift(index-1, originalRoster, size);

		if(dev_getColumn(index, size) == size-1)
			right = FREE;
		else
			right = dev_getOrigRosterShift(index+1, originalRoster, size);
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))||(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= settings.mphPenalization;
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))&&(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= settings.mphPenalization;
	}
	
	if(settings.isLeftRightConstrainSoft)
	{
		result+= (dev_isLeftHardConstrainBroken(index, size, shiftType, currentRoster, erRoster, settings) + dev_isRightHardConstrainBroken(index, size, shiftType, currentRoster))*settings.lrcsPenalization;
	}
	
	if(settings.isNursePreferences)
		result += dev_getNursePreference(index, shiftType, nursePreferences);
	if(settings.isMulticommodityFlowConstrins)
	{
		if(shiftType == NIGHT)
			return result;
		if(dev_getColumn(index, size) == 0)
			left = FREE;
		else
			left = dev_getRosterShift(index-1, currentRoster, size);
		if(dev_getColumn(index,size) == size - 1)
			right = FREE;
		else
			right = dev_getRosterShift(index+1, currentRoster, size);

		if(shiftType == PM)
		{
			conCount = 1;
			if(left == PM)
			{
				while((index >= i)&&(dev_getRosterShift(index-i, currentRoster, size)==PM))
				{
					if(dev_getRow(index-i, size) == dev_getRow(index, size))
						i++;
					else
						break;
				}
				conCount = i;
			}
			if(right == PM)
			{
				i = 2;
				while(((index + i)<=nurseCount*size)&&(dev_getRosterShift(index+i, currentRoster, size)==PM))
				{
					if(dev_getRow(index-i, size) == dev_getRow(index, size))
						i++;
					else
						break;
				}
				conCount += i - 1;
			}
			if(conCount > 1)
				result += (conCount-1)*settings.mfcNightPenalization;

		}
		if(shiftType == AM)
		{	
			conCount = 1;
			if(left == AM)
			{
				while((index >= i)&&(dev_getRosterShift(index-i, currentRoster, size)==AM))
				{
					if(dev_getRow(index-i, size) == dev_getRow(index, size))
						i++;
					else
						break;
				}
				conCount = i;
			}
			if(right == AM)
			{
				i = 2;
				while(((index + i)<=nurseCount*size)&&(dev_getRosterShift(index+i, currentRoster, size)==AM))
				{
					if(dev_getRow(index-i, size) == dev_getRow(index, size))
						i++;
					else
						break;
				}
				conCount += i - 1;
			}
			if(conCount > 3)
				result += (conCount-3)*settings.mfcPmPenalization;
		}
	}
	return result;
}







//Heterogenous
__global__ void kernel(unsigned int* index, int size, unsigned int nurseCount, unsigned int* origRoster, unsigned int* shiftMap, unsigned int* isFirst, unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings, unsigned int* result)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	//unsigned int numInstance = tid/(size*nurseCount);
	//unsigned int nurse = tid%(size*nurseCount);
	unsigned int numInstance = tid/nurseCount;
	unsigned int nurse = tid%nurseCount;
	unsigned int offset = numInstance*nurseCount;

	result[numInstance*nurseCount + nurse] = dev_CompatibilityPenalization(nurse*size + (index+offset)[nurse]%size ,size, nurseCount, dev_getOrigRosterShift((index+offset)[nurse], origRoster, size), origRoster, shiftMap+offset, isFirst[numInstance], currentRoster+2*offset, freePreferenceMap, nursePreferences, sdscFreeDays, erRoster, erShiftMap, settings);
	//result[numInstance*nurseCount + nurse] = dev_CompatibilityPenalization(0, size, nurseCount, 0, shiftMap, isFirst[0], currentRoster, freePreferenceMap);


	//void nurseCompatibilityPenalization(int size, unsigned int* actualShifts, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* shiftMap, unsigned int isFirst, int nurse)
	//nurseCompatibilityPenalization(_dayCount, actualShifts + offset, currentRoster + 2*offset, penalityFunctions + offset, shiftMap + offset, firstPhase[numInstance], i);
	//unsigned int CompatibilityPenalization(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster)
	//penalityFunctions[nurse] = CompatibilityPenalization(nurse*_dayCount+actualShifts[nurse]%_dayCount, size, getOrigRosterShift(actualShifts[nurse]), shiftMap, isFirst, currentRoster);
	//ui dev_CompatibilityPenalization(unsigned int index, int size, unsigned int nurseCount, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster, unsigned int* freePreferenceMap)
}

unsigned int deviceCompatibilityPenalizationHost(unsigned int* index, int size, unsigned int* shiftMap, unsigned int* isFirst,unsigned int* currentRoster, unsigned int* penaltyFunction)
{
	cudaMemcpy( dev_shiftMap, shiftMap, sizeof(int)*(_nurseCount)*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_currentRoster, currentRoster, sizeof(int)*(2*_nurseCount)*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_index, index, _paralelInstanceCount * _nurseCount * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_origRoster, _originalRoster, sizeof(int)*_nurseCount*2, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_isFirst, isFirst, _paralelInstanceCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_freePreferenceMap, _freePreferenceMap, sizeof(int)*_nurseCount, cudaMemcpyHostToDevice);
	kernel<<<_paralelInstanceCount,_nurseCount>>>(dev_index, size, _nurseCount, dev_origRoster, dev_shiftMap, dev_isFirst, dev_currentRoster, dev_freePreferenceMap, dev_nursePreferences, dev_sdscFreeDays, dev_erRoster, dev_erShiftMap, _settings, dev_result);
	cudaMemcpy( penaltyFunction, dev_result, sizeof(int)*_nurseCount*_paralelInstanceCount, cudaMemcpyDeviceToHost);
	return 1;
}














// Homogenous

const int instancesPerBlock = 2;
const int maxNurses = 32;

__global__ void homogenousKernel(unsigned int* shiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestUtility, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int* bestLock, unsigned int* absenceArray, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, unsigned int* localSearch, unsigned int* runsWithoutSucess, unsigned int* runsWithSameProbability, float* probability, unsigned int* localSearchLock, unsigned int* exit, unsigned int* run, unsigned int runCount, Settings settings, unsigned int nonFreeDays, unsigned int initialChangeCount, unsigned int nurseCount, unsigned int dayCount, unsigned int fairnessAverage, curandState* state, int seed)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned int numInstance = tid/nurseCount;
	unsigned int nurse = tid%nurseCount;
	unsigned int offset = numInstance*nurseCount;
	numInstance = threadIdx.x/nurseCount;

	//cuPrintf("init: %d, %d, %d\n", tid, blockIdx.x, numInstance);
	//result[numInstance*nurseCount + nurse] = dev_CompatibilityPenalization(nurse*size + (index+offset)[nurse]%size, size, nurseCount, dev_getOrigRosterShift((index+offset)[nurse], origRoster, size), shiftMap+offset, isFirst[numInstance], currentRoster+2*offset, freePreferenceMap, nursePreferences, sdscFreeDays, erRoster, erShiftMap, settings);
	curand_init(seed, tid, 0, &state[tid]);
	dev_KernelRun(shiftList + offset*dayCount, shiftMap + offset, shiftMapInit, originalRoster, currentRoster + 2*offset, currentRosterInit, bestUtility, bestRoster, bestShiftList, bestLock, absenceArray, penalityFunctions + offset, nonAssignedShifts + offset, staffGaps + 3*dayCount*(tid/nurseCount), staffGapsInit, freePreferenceMap, nursePreferences, sdscFreeDays, erRoster, erShiftMap, localSearch, runsWithoutSucess, runsWithSameProbability, localSearchLock, probability, exit, run, runCount, settings, nonFreeDays, initialChangeCount, nurseCount, dayCount, fairnessAverage, nurse, numInstance, &state[tid]);
}

//Method for rerostering - CPU version of GPU algorithm
__device__ __inline void dev_KernelRun(unsigned int* shiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestUtility, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int* bestLock, unsigned int* absenceArray, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, unsigned int* localSearch, unsigned int* runsWithoutSucess, unsigned int* runsWithSameProbability, unsigned int* localSearchLock, float* probability, unsigned int* exit, unsigned int* run, unsigned int runCount, Settings settings, unsigned int nonFreeDays, unsigned int initialChangeCount, unsigned int nurseCount, unsigned int dayCount, unsigned int fairnessAverage, unsigned int nurse, unsigned int instanceInBlockIndex, curandState* state)
{
	unsigned int i = 0;
	unsigned int min = INTMAX, minCount = 0;
	int minIndex = -1;
	__shared__ unsigned int utility[instancesPerBlock];
	__shared__ unsigned int infeasible[instancesPerBlock];
	__shared__ unsigned int actualShifts[maxNurses*instancesPerBlock];
	__shared__ unsigned int shiftSecondPhase[instancesPerBlock];
	__shared__ unsigned int saveBestFlag[instancesPerBlock];
	__shared__ unsigned int keepShiftList[instancesPerBlock];
	__shared__ unsigned int backTrackCounter[instancesPerBlock];
	unsigned int offset = nurseCount*((int)(threadIdx.x/nurseCount));
	__shared__ unsigned int index[instancesPerBlock];
	__shared__ unsigned int firstPhase[instancesPerBlock];

	if(threadIdx.x % nurseCount == 0)
	{
		index[instanceInBlockIndex] = dayCount;
		firstPhase[instanceInBlockIndex] = 1;
		infeasible[instanceInBlockIndex] = 0;
		saveBestFlag[instanceInBlockIndex] = 0;
		backTrackCounter[instanceInBlockIndex] = 0;
		keepShiftList[instanceInBlockIndex] = 0;
	}
	__syncthreads();
	while(1)
	{
		//PartA
		if((index[instanceInBlockIndex] == dayCount)||infeasible[instanceInBlockIndex]) // Inicializace
		{
			if(threadIdx.x % nurseCount == 0)
			{
				//cuPrintf("%d/%d - jsem tu\n", *run, runCount);
				if((*run) >= runCount)
				{
					*exit = 1;
				}
				saveBestFlag[instanceInBlockIndex] = 0;
				infeasible[instanceInBlockIndex] = 0;
				index[instanceInBlockIndex] = 0;
				firstPhase[instanceInBlockIndex] = 1;
				if(keepShiftList[instanceInBlockIndex])
				{
					backTrackCounter[instanceInBlockIndex]++;
					if(backTrackCounter[instanceInBlockIndex] > 0)// 10*dayCount)
					{
						//cuPrintf("c%d\n", *run);
						backTrackCounter[instanceInBlockIndex] = 0;
						keepShiftList[instanceInBlockIndex] = 0;
					}
				}
				else
				{
					backTrackCounter[instanceInBlockIndex] = 0;
				}
			}
			dev_nurseInitialize(currentRoster, currentRosterInit, bestShiftList, shiftMap, shiftMapInit, shiftList, nonAssignedShifts, staffGaps, staffGapsInit, *localSearch, nonFreeDays, *probability, nurseCount, dayCount, nurse, keepShiftList[instanceInBlockIndex], state);
			if(threadIdx.x % nurseCount == 0)
			{
				keepShiftList[instanceInBlockIndex] = 0;
			}
		}

		__syncthreads(); // Synchronizace - index, firstPhase - inicializuje nurse0, používají všechny
		if(firstPhase[instanceInBlockIndex]) // Urèení indexu služby pro pøiøazení
		{
			dev_nurseLoadFirstPhase(index[instanceInBlockIndex], shiftList, actualShifts + offset, dayCount, nurse);
		}
		else
		{
			dev_nurseLoadSecondPhase(shiftSecondPhase[instanceInBlockIndex], actualShifts + offset, nurse);
		}
		__syncthreads(); // - Pravdìpodobnì není tøeba - jedinná zmìna je actualshifts a to nám staèí pro danou sestru.

		//PartB
		if(*exit) // Výpoèetní èást
		{
			return;
		}
		
		dev_nurseCompatibilityPenalization(nurseCount, dayCount, actualShifts + offset, originalRoster, currentRoster, penalityFunctions, shiftMap, firstPhase[instanceInBlockIndex], freePreferenceMap, nursePreferences, sdscFreeDays, erRoster, erShiftMap, settings, dayCount, nurse);
		__syncthreads(); // Synchronizace nutná pro fázi 2 - sestra 0 musí znát penalty function od všech sester

		//PartC
		minIndex = -1;
		min = INTMAX;
		minCount = 0;

		if(firstPhase[instanceInBlockIndex]) // Pøiøazení služby do rozvrhu
		{
			dev_nurseAssign(actualShifts + offset, penalityFunctions, originalRoster, currentRoster, nonAssignedShifts, shiftMap, shiftMapInit, absenceArray, staffGaps, dayCount, nurse);
		}
	
		if(threadIdx.x % nurseCount == 0)
		{
			if(!firstPhase[instanceInBlockIndex]) // Nalezení sestry s minimální penalty function
			{
				for(i = 0; i < nurseCount; i++)
				{
					if((penalityFunctions[i] < min))// &&(getOrigRosterShift(i*_dayCount + shiftSecondPhase%_dayCount)!=getOrigRosterShift(shiftSecondPhase)))
					{
						min = penalityFunctions[i];
						minIndex = i;
						minCount = 1;
					}	
					else if((min != INTMAX)&&(penalityFunctions[i] == min))
					{
						minCount++;
						if(curand_uniform(state) < 1/((float)minCount))
						{
							minIndex = i;
						}
					}
				}
			}
			
			if(!firstPhase[instanceInBlockIndex]) // Pøiøazení službì s min. penalty function | infeasible
			{
				if(minIndex > -1)
				{
					dev_setRosterShift(currentRoster, minIndex*dayCount + (actualShifts+offset)[minIndex]%dayCount, dev_getOrigRosterShift(shiftSecondPhase[instanceInBlockIndex], originalRoster, dayCount), dayCount);
					dev_setBitOfBitArray(shiftMap, minIndex*dayCount + (actualShifts + offset)[minIndex]%dayCount, 1, dayCount);
				}
				else
				{
					infeasible[instanceInBlockIndex] = 1;
					if(index[instanceInBlockIndex] > 0){
						int swapTmp,  swapNurse;
						keepShiftList[instanceInBlockIndex] = 1;
						swapNurse = ((actualShifts + offset)[0])/dayCount;
						swapTmp = shiftList[swapNurse*dayCount+index[instanceInBlockIndex]];
						shiftList[swapNurse*dayCount+index[instanceInBlockIndex]] = shiftList[swapNurse*dayCount+index[instanceInBlockIndex]-1];
						shiftList[swapNurse*dayCount+index[instanceInBlockIndex]-1] = swapTmp;
					}
					atomicAdd(run, 1);
				}
			}
		}
		
		__syncthreads(); // Synchronizace - sestra 0 musí znát nonAssignedShifts všech sester
		if(threadIdx.x % nurseCount == 0)
		{
			minIndex = -1;
			for(i = 0; i < nurseCount; i++) // Hledání nepøiøazené služby
			{
				if(nonAssignedShifts[i] != INTMAX)
				{
					minIndex = i;
					break;
				}
			}
			if(minIndex == -1) // Neexistuje-li nepøiøazená služba => 1.fáze
			{
				firstPhase[instanceInBlockIndex] = 1;
				index[instanceInBlockIndex]++;
			}
			else // jinak urèi danou službu a 2.fáze
			{
				firstPhase[instanceInBlockIndex] = 0;
				shiftSecondPhase[instanceInBlockIndex] = nonAssignedShifts[minIndex];
				nonAssignedShifts[minIndex] = INTMAX;
			}
		}
		
		__syncthreads(); // Synchronizace kvùli index
		if((index[instanceInBlockIndex] == dayCount)&&(!infeasible[instanceInBlockIndex]))
		{
			penalityFunctions[nurse] = dev_nurseUtilityFunctionMoz(currentRoster, originalRoster, nursePreferences, dayCount, settings, fairnessAverage, nurse);
			//cuPrintf("s: %d, %d, %d\n", instanceInBlockIndex, penalityFunctions[nurse], nurse);
		}
		__syncthreads(); // Synchronizace - nutno znát penalty function pro výpoèet utility

		
		if((index[instanceInBlockIndex] == dayCount)&&(!infeasible[instanceInBlockIndex]))
		{
			if(threadIdx.x % nurseCount == 0)
			{
				utility[instanceInBlockIndex] = 0;
				for(i = 0; i < nurseCount; i++)
				{
					utility[instanceInBlockIndex] += penalityFunctions[i];
					//cuPrintf("%d, %d, %d, %d\n", utility[instanceInBlockIndex], instanceInBlockIndex, penalityFunctions[i], i);
				}
				utility[instanceInBlockIndex]-=initialChangeCount*settings.changePenalization;
				atomicAdd(run,1);
				//*run = *run + 1;
				i = 1;
				while(i)
				{
					if (atomicExch(bestLock, 1) == 0) {
						if(*bestUtility > utility[instanceInBlockIndex])
						{
							saveBestFlag[instanceInBlockIndex] = 1;
							//cuPrintf("best i: %d, v:%d\n", instanceInBlockIndex, utility[instanceInBlockIndex]);
							*bestUtility = utility[instanceInBlockIndex];
						}
						i = 0;
						atomicExch(bestLock,0);
					}
				}
				if(*bestUtility > utility[instanceInBlockIndex])
				{
					i = 1;
					while(i)
					{
						if (atomicExch(localSearchLock, 1) == 0) {
							runsWithoutSucess = 0;
							i = 0;
							atomicExch(localSearchLock,0);
						}
					}
				}
			}
		}

		__syncthreads(); // Synchronizace - všechny sestry musí znát saveBestFlag, které nastavuje sestra 0
		if(saveBestFlag[instanceInBlockIndex])
		{
			i = 1;
			while(i)
			{
				if (atomicExch(bestLock, 1) == 0) {
					for(minIndex = 0; minIndex < nurseCount; minIndex++)
						dev_nurseSaveBest(currentRoster, shiftList, bestRoster, bestShiftList, dayCount, minIndex);
					i = 0;
					atomicExch(bestLock,0);
				}
			}
		}
		else
		{
			if((threadIdx.x % nurseCount == 0)&&((index[instanceInBlockIndex] == dayCount)||(infeasible[instanceInBlockIndex])))
			{
				//cuPrintf("%d, tuuuu\n", nurse);
				i = 1;
				while (i) {
					if (atomicExch(localSearchLock, 1) == 0) {
						if(!localSearch)
						{
							if(*bestUtility != INTMAX)
								*runsWithoutSucess++;
							if(*runsWithoutSucess > LOCAL_SEARCH_RWS_TRESHOLD)
							{	
								*localSearch=1;
								*probability = 0.25;
								*runsWithoutSucess = 0;
							}
						}
						else
						{
							*runsWithSameProbability++;
							if(*runsWithSameProbability > LOCAL_SEARCH_RSP_TRESHOLD)
							{
								*probability /= 2;
							}
							if(*probability < 0.001F)
							{
								*localSearch = 0;
								*runsWithSameProbability = 0;
							}
						}
						i = 0;
						atomicExch(localSearchLock,0);
					}
				} 
			}
		}
		__syncthreads();// - není tøeba(jen v pøípadì seriového ukládání nejlepšího rozvrhu - nesmí se pøepsat currentRoster dokud jej sestra 0 neuloží
	}
	
}

//Method for nurse to save actual roster row to storage for best one
__device__ __inline void dev_nurseSaveBest(unsigned int* currentRoster, unsigned int* shiftList, unsigned int* bestRoster, unsigned int* bestShiftList, unsigned int dayCount, unsigned int nurse)
{
	unsigned int i;
	bestRoster[2*nurse] = currentRoster[2*nurse];
	bestRoster[2*nurse+1] = currentRoster[2*nurse+1];
	for(i = 0; i < dayCount; i++)
		bestShiftList[nurse*dayCount + i] = shiftList[nurse*dayCount + i];
}

//Method for nurse to assign
__device__ __inline void dev_nurseAssign(unsigned int* actualShifts, unsigned int* penalityFunction, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* nonAssignedShifts, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* absenceArray, unsigned int* staffGaps, unsigned int dayCount, unsigned int nurse)
{
	if((!dev_getBitOfBitArray(shiftMapInit, actualShifts[nurse], dayCount))||(dev_getBitOfBitArray(absenceArray, actualShifts[nurse], dayCount)))
	{
		if(dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount)!=FREE)
		{
			if(penalityFunction[nurse] != INTMAX)
			{
				dev_setRosterShift(currentRoster, actualShifts[nurse], dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount), dayCount);
				dev_setBitOfBitArray(shiftMap, actualShifts[nurse], 1, dayCount);
			}
			else
			{
				if(staffGaps[3*dev_getColumn(actualShifts[nurse], dayCount)+dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount)-1] > 0)
				{
					if(atomicSub(&staffGaps[3*dev_getColumn(actualShifts[nurse], dayCount)+dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount)-1], 1) - 1 > maxNurses)
					{	
						staffGaps[3*dev_getColumn(actualShifts[nurse], dayCount)+dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount)-1] = 0;
						nonAssignedShifts[nurse] = actualShifts[nurse];
					}
				}
				else
				{
					//printRoster(currentRoster);
					//printf("%d %d %d \n", actualShifts[nurse]%_dayCount, nurse, actualShifts[nurse]);
					nonAssignedShifts[nurse] = actualShifts[nurse];
					//nurseCompatibilityPenalization(_dayCount, actualShifts, currentRoster, penalityFunction, shiftMap, 1, nurse);
				}
			}
		}
	}
}

__device__ __inline void dev_nurseCompatibilityPenalization(unsigned int nurseCount, int size, unsigned int* actualShifts, unsigned int* originalRoster, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* shiftMap, unsigned int isFirst, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings, unsigned int dayCount, unsigned int nurse)
{
	penalityFunctions[nurse] =  dev_CompatibilityPenalizationFlat(nurse*dayCount+actualShifts[nurse]%dayCount, size, nurseCount, dev_getOrigRosterShift(actualShifts[nurse], originalRoster, dayCount), originalRoster, shiftMap, isFirst, currentRoster, freePreferenceMap, nursePreferences, sdscFreeDays, erRoster, erShiftMap, settings);
}

__device__ __inline void dev_nurseLoadSecondPhase(unsigned int shiftSecondPhase, unsigned int* actualShifts, unsigned int nurse)
{
	actualShifts[nurse] = shiftSecondPhase;
}

__device__ __inline void dev_nurseLoadFirstPhase(int index, unsigned int* shiftList, unsigned int* actualShifts, unsigned int dayCount, unsigned int nurse)
{
	actualShifts[nurse] = shiftList[nurse*dayCount+index];
}

__device__ __inline void dev_nurseInitialize(unsigned int* currentRoster, unsigned int* currentRosterInit, unsigned int* bestShiftList, unsigned int* shiftMap, unsigned int* shiftMapInit, unsigned int* shiftList, unsigned int* nonAssignedShift, unsigned int* staffGaps, unsigned int* staffGapsInit, unsigned int isLocal, unsigned int nonFreeDays, float probability, unsigned int nurseCount, unsigned int dayCount, unsigned int nurse, unsigned int keepShiftList, curandState* state)
{
	unsigned int i;
	int swap, swapIndex;
	nonAssignedShift[nurse] = INTMAX;
	currentRoster[2*nurse] = currentRosterInit[2*nurse];
	currentRoster[2*nurse+1] = currentRosterInit[2*nurse+1];
	shiftMap[nurse] = shiftMapInit[nurse];
	for(i = nurse; i < dayCount; i+=nurseCount)
	{
		staffGaps[i*3] = staffGapsInit[i*3];
		staffGaps[i*3+1] = staffGapsInit[i*3+1];
		staffGaps[i*3+2] = staffGapsInit[i*3+2];
	}
	if(keepShiftList == 0)
	{
		if(isLocal)
		{
			for(i = 0; i < dayCount; i++)
			{	
				shiftList[nurse*dayCount + i] = bestShiftList[nurse*dayCount + i];
			}
		}
		for(i = 0; i < nonFreeDays; i++)
		{
			if((!isLocal)||(curand_uniform(state) < probability))
			{
				//swapIndex = nonFreeDays - (unsigned int)(curand_uniform(state)*nonFreeDays);
				swapIndex = (unsigned int)(((1.F-(float)curand_uniform(state))*0.9999)*nonFreeDays);
				if((swapIndex < 0)||(swapIndex >= nonFreeDays))
					cuPrintf("error - %d\n", swapIndex);
			}
			else
			{
				swapIndex = i;
			}
			swapIndex = nurse*dayCount+swapIndex;
			swap = shiftList[swapIndex];
			shiftList[swapIndex] = shiftList[nurse*dayCount + i];
			shiftList[nurse*dayCount + i] = swap;
		}
	}
}

__device__ __inline int dev_minShiftTypeConsecutiveConstrainTest(int actual, int NIGHTs, int AMs, int PMs, Settings settings)
{
	if((actual != NIGHT)&&(NIGHTs != 0)&&(NIGHTs < (int)settings.mmhcMinShiftTypeConsecutive))
		return 1;
	if((actual != AM)&&(AMs != 0)&&(AMs < (int)settings.mmhcMinShiftTypeConsecutive))
		return 1;
	if((actual != PM)&&(PMs != 0)&&(PMs < (int)settings.mmhcMinShiftTypeConsecutive))
		return 1;
	return 0;
}

// Calculate utility function of roster
__device__ unsigned int dev_nurseUtilityFunction(unsigned int* currentRoster, unsigned int* originalRoster, unsigned int* nursePreference, unsigned int dayCount, Settings settings, unsigned int fairnessAverage, unsigned int nurse)
{
	unsigned int i;
	int result = 0;
	int shift;
	int NIGHTs = 0, AMs = 0, PMs = 0, FREEs = 0;
	unsigned int countNIGHT = 0, countAM = 0, countNight = 0;
	unsigned int dutyCount = 0, consecutiveDutyCount = 0;
	for(i = 0; i < dayCount; i++)
	{
		shift = dev_getRosterShift(nurse*dayCount+i, currentRoster, dayCount);
		if(settings.isNursePreferences)
			result += dev_getNursePreference(nurse*dayCount+i, shift, nursePreference);
		if(shift != FREE)
			dutyCount++;
		if(shift != dev_getOrigRosterShift(nurse*dayCount+i, originalRoster, dayCount))
			result += settings.changePenalization;
		if(i == 0)
			AMs = PMs = FREEs = NIGHTs = 0;
		if(shift == NIGHT)
		{
			if((settings.isMinMaxHardConstrain)&&(dev_minShiftTypeConsecutiveConstrainTest(NIGHT, NIGHTs, AMs, PMs,settings)))
				result += settings.softInfeasiblePenalization;
			AMs = PMs = FREEs = 0;
			consecutiveDutyCount++;
			NIGHTs++;
			countNIGHT++;
		}
		else if(shift == AM)
		{
			if((settings.isMinMaxHardConstrain)&&(dev_minShiftTypeConsecutiveConstrainTest(AM, NIGHTs, AMs, PMs,settings)))
				result += settings.softInfeasiblePenalization;
			PMs = FREEs = NIGHTs = 0;
			consecutiveDutyCount++;
			AMs++;
			countAM++;
			if(AMs > 3)
				result += settings.mfcPmPenalization;
		}
		else if(shift == PM)
		{
			if((settings.isMinMaxHardConstrain)&&(dev_minShiftTypeConsecutiveConstrainTest(PM, NIGHTs, AMs, PMs, settings)))
				result += settings.softInfeasiblePenalization;
			AMs = FREEs = NIGHTs = 0;
			consecutiveDutyCount++;
			PMs++;
			countNight++;
			if(PMs > 1)
				result += settings.mfcNightPenalization;
		}
		else if(shift == FREE)
		{
			if((settings.isMinMaxHardConstrain)&&(consecutiveDutyCount!=0)&&(consecutiveDutyCount < settings.mmhcMinConsecutive))
				result += settings.softInfeasiblePenalization;
			if((settings.isMinMaxHardConstrain)&&(dev_minShiftTypeConsecutiveConstrainTest(FREE, NIGHTs, AMs, PMs, settings)))
				result += settings.softInfeasiblePenalization;
			AMs = PMs = NIGHTs = 0;
			FREEs++;
			consecutiveDutyCount = 0;
			if(FREEs > 3)
				result += settings.mfcFreePenalization;
		}
	}
	if(!settings.isSevenDaySequenceConstrain)
	{
		result += settings.fairnessPenalization*((dutyCount - fairnessAverage)>0?(dutyCount-fairnessAverage):(fairnessAverage-dutyCount));
	}
	if((settings.isMinMaxHardConstrain)&&(dutyCount < settings.mmhcMinWorking))
		result += settings.softInfeasiblePenalization;
	if((settings.isMinMaxHardConstrain)&&(countNIGHT < settings.mmhcMinShiftType)&&(countAM < settings.mmhcMinShiftType)&&(countNight < settings.mmhcMinShiftType))
		result += settings.softInfeasiblePenalization;
	return result;
}

__device__ unsigned int dev_nurseUtilityFunctionMoz(unsigned int* currentRoster, unsigned int* originalRoster, unsigned int* nursePreference, unsigned int dayCount, Settings settings, unsigned int fairnessAverage, unsigned int nurse)
{
	unsigned int i;
	int result = 0;
	for(i = 0; i < dayCount; i++)
	{
		if(dev_getRosterShift(nurse*dayCount+i, currentRoster, dayCount) != dev_getOrigRosterShift(nurse*dayCount+i, originalRoster, dayCount))
			result += settings.changePenalization;
	}
	return result;
}

__device__ unsigned int dev_CompatibilityPenalizationFlat(unsigned int index, int size, unsigned int nurseCount, unsigned int shiftType, unsigned int* originalRoster, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster, unsigned int* freePreferenceMap, unsigned int* nursePreferences, unsigned int* sdscFreeDays, unsigned int* erRoster, unsigned int* erShiftMap, Settings settings)
{
	unsigned int result = 0;
	unsigned int left, right;
	unsigned int shifts, shift, frees, nurse, shiftMapNurse;
	int tmp = 0;
	int i, startDay, day;

	if(dev_getBitOfBitArray(shiftMap, index, size))
	{
		return INTMAX;
	}
	
	// isLeftHardConstrin Broken
	if(dev_getColumn(index, size) == 0)
		left = dev_getExtendedRosterShift(dev_getRow(index, size), erRoster);
	else
		left = dev_getRosterShift(index-1, currentRoster, size);
	if(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT)))
		return INTMAX;


	//isRightHardConstrainBroken
	if(dev_getColumn(index, size) != size-1)
		right = dev_getRosterShift(index+1, currentRoster, size);
	else
		right = FREE;
	if((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))
		return INTMAX;

	if(settings.isMPHeuristic == 1)
	{
		result += 2*settings.mphPenalization;
		if((left!=FREE)||(right!=FREE))
			result -= settings.mphPenalization;
		if((left!=FREE)&&(right!=FREE))
			result -= settings.mphPenalization;
	}

	if(settings.isMPHeuristic == 2)
	{
		result += 2*settings.mphPenalization;
		if(dev_getColumn(index, size) == 0)
			left = dev_getExtendedRosterShift(dev_getRow(index, size), erRoster);
		else
			left = dev_getOrigRosterShift(index-1, originalRoster, size);

		if(dev_getColumn(index, size) == size-1)
			right = FREE;
		else
			right = dev_getOrigRosterShift(index+1, originalRoster, size);
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))||(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= settings.mphPenalization;
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))&&(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= settings.mphPenalization;
	}

	day = dev_getColumn(index, size);
	nurse = dev_getRow(index, size);
	shiftMapNurse = shiftMap[nurse];
	shiftMapNurse = shiftMapNurse&(~freePreferenceMap[nurse]);
	shifts = 0;
	frees = 0;
	if(settings.isExtendedRoster)
		startDay = day-6;	
	else
		startDay = day>5?day-6:0;
	for(i = startDay; i < startDay+7; i++)
	{
		if(i >= size)
		{
			frees++;
			shifts = shifts<<1;
		}
		else
		{
			if(i != day)
			{
				if(i < 0)
				{
					shift = (dev_getBitOfBitArray(erShiftMap, nurse*size-i, size))?1:0;
				}
				else
				{
					shift = (dev_getBitOfBitArray(&shiftMapNurse, i, size))?1:0;
				}
			}
			else
			{
				shift = (shiftType==FREE)?0:1;
			}
			shifts += shift;
			shifts = shifts << 1;
			if(!shift)
			{
				frees++;
			}
		}
	}
	if(frees < sdscFreeDays[nurse])
	{
		return INTMAX;
	}
	for(i = startDay+7; i<(int)day+7; i++) // Check all other seven-day sequencies
	{
		if(i < (int)size)
			shift = (dev_getBitOfBitArray(&shiftMapNurse, i, size))?1:0;
		else
			break;
		if(!shift) frees++;
		if(!((shifts>>(7-tmp))&1))
			frees--;
		if(frees < sdscFreeDays[nurse])
			return INTMAX;
		tmp++;
	}

	if(isFirst)
		return 0;
	return result;
}

void homogenousKernelCall()
{
	struct timeb startTime;
	struct timeb endTime;
	ftime(&startTime);

	unsigned int i, j;
	unsigned int localSearch = 0;
	unsigned int runsWithoutSucess = 0;
	float probability = 1;
	unsigned int runsWithSameProbability = 0;
	unsigned int bestLock = 0;
	unsigned int localSearchLock = 0;
	unsigned int run = 0;
	unsigned int exit_code = 0;

	for(i = 0; i < _shiftToReroster; i++)
	{
		_shiftList[i] = i;
	}

	for(i = 0; i < _nurseCount; i++)
	{
		for(j = 0; j < _dayCount; j++)
		{
			if((getOrigRosterShift(_shiftList[i*_dayCount+j]) == FREE)||((getBitOfBitArray(_shiftMapInit, _shiftList[i*_dayCount+j]))&&(!getBitOfBitArray(_absenceArray, _shiftList[i*_dayCount+j]))))
			{
				int k = (i+1)*_dayCount-1;
				int tmp;
				while(((getOrigRosterShift(_shiftList[k])==FREE)||((getBitOfBitArray(_shiftMapInit, _shiftList[k]))&&(!getBitOfBitArray(_absenceArray, _shiftList[k]))))&&(!(k==i*_dayCount+j)))
					k--;
				if(((k%_dayCount) < _nonFreeDays)||(k==i*_dayCount+j))
					break;
				tmp = _shiftList[k];
				_shiftList[k] = _shiftList[i*_dayCount+j];
				_shiftList[i*_dayCount+j] = tmp;
			}
		}
	}

	for(i = 1; i <_paralelInstanceCount; i++) // Copy shift list of first instance to others instances
	{
		for(j = 0; j < _shiftToReroster; j++)
			_shiftList[i*_shiftToReroster + j] = _shiftList[j];
	}

	cudaPrintfInit();

	cudaMemcpy( dev_shiftList, _shiftList, sizeof(int)*_nurseCount*_dayCount*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_shiftMapInit, _shiftMapInit, sizeof(int)*_nurseCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_currentRosterInit, _currentRosterInit, sizeof(int)*2*_nurseCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_bestUtility, &_bestUtility, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_bestRoster, _bestRoster, sizeof(int)*2*_nurseCount, cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_bestShiftList, _bestShiftList, sizeof(int)*_nurseCount*_dayCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_bestLock, &bestLock, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_absenceArray, _absenceArray, sizeof(int)*_nurseCount, cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_penalityFunctions, _penalityFunctions, sizeof(int)*_nurseCount*_paralelInstanceCount, cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_nonAssignedShifts, _nonAssignedShifts, sizeof(int)*_nurseCount*_paralelInstanceCount, cudaMemcpyHostToDevice);
	//cudaMemcpy( dev_staffGaps, dev_staffGaps, sizeof(int)*3*_dayCount*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_staffGapsInit, _staffGapsInit, sizeof(int)*3*_dayCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_localSearch, &localSearch, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_runsWithoutSucess, &runsWithoutSucess, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_runsWithSameProbability, &runsWithSameProbability, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_localSearchLock, &localSearchLock, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_exit, &exit_code, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_run, &run, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_probability, &probability, sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy( dev_shiftMap, _shiftMap, sizeof(int)*(_nurseCount)*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_currentRoster, _currentRoster, sizeof(int)*(2*_nurseCount)*_paralelInstanceCount, cudaMemcpyHostToDevice);
	cudaMemcpy( dev_freePreferenceMap, _freePreferenceMap, sizeof(int)*_nurseCount, cudaMemcpyHostToDevice);

	homogenousKernel<<<_paralelInstanceCount/instancesPerBlock,_nurseCount*instancesPerBlock>>>(dev_shiftList, dev_shiftMap, dev_shiftMapInit, dev_origRoster, dev_currentRoster, dev_currentRosterInit, dev_bestUtility, dev_bestRoster, dev_bestShiftList, dev_bestLock, dev_absenceArray, dev_penalityFunctions, dev_nonAssignedShifts, dev_staffGaps, dev_staffGapsInit, dev_freePreferenceMap, dev_nursePreferences, dev_sdscFreeDays, dev_erRoster, dev_erShiftMap, dev_localSearch, dev_runsWithoutSucess, dev_runsWithSameProbability, dev_probability, dev_localSearchLock, dev_exit, dev_run, _runCount, _settings, _nonFreeDays, _initialChangeCount, _nurseCount, _dayCount, _fairnessAverage, dev_states, (int)time(NULL));
	cudaMemcpy(_bestRoster, dev_currentRoster, sizeof(int)*2*_nurseCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_bestUtility, dev_bestUtility, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(_bestRoster, dev_bestRoster, sizeof(int)*2*_nurseCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(&run, dev_run, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(_shiftMap, dev_shiftMap, sizeof(int)*_nurseCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(_shiftList, dev_shiftList, sizeof(int)*_nurseCount*_dayCount, cudaMemcpyDeviceToHost);
	ftime(&endTime);
		printf("Time: %f\n", endTime.time - startTime.time + (endTime.millitm - startTime.millitm)/(float)1000);
	cudaPrintfDisplay();
	cudaPrintfEnd();
	/*for(i = 0; i < _nurseCount; i++)
	{
		printf("%2d: ", _dayCount*i);
		for(j = 0; j < _dayCount; j++)
			printf("%3d, ", _shiftList[_dayCount*i + j]);
		printf("\n");
	}*/
	printf("%s \n", cudaGetErrorString(cudaGetLastError()));
}