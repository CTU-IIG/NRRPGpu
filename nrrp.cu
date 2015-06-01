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

unsigned int* _originalRoster; // Original roster array
unsigned int _nurseCount = 0; // Number of nurses  for rerostering
unsigned int _dayCount = 0; // Number of days for rerostering
unsigned int _shiftToReroster = 0; // Number of shifts for rerostering
unsigned int _runCount = 1; // Number of runs
unsigned int _shiftChangeMaximum = 16; // Maximal number of changes
unsigned int _initialChangeCount = 0; // Number of rerostered shifts
unsigned int _paralelInstanceCount = 1200;//960;
unsigned int _absenceCount; // Number of absences
unsigned int _bestUtility = INTMAX; // Best utility function
unsigned int* _bestRoster; // List of changes of best roster
unsigned int* _shiftList; // Ordered list of shift to reroster
unsigned int* _shiftMap; // Bit array for mapping already rostered shifts
unsigned int* _shiftMapInit; // Initialization array for bit array for mapping already rostered shifts
unsigned int* _freePreferenceMap;
unsigned int* _absenceArray; // Bit array of absences
unsigned int* _currentRoster; // Actual roster
unsigned int* _currentRosterInit; // Initial array for actual rosters
unsigned int* _penaltyFunctions; // Value of penalty for every try of assign at turn
unsigned int* _penaltyFunctionsTest; // -//- for GPU correctness testing
unsigned int* _nonAssignedShifts;
unsigned int _nonFreeDays; // Number of days, when any nurse has some shift
unsigned int _fairnessAverage = 0; // Number of duties averaged over the nurses
unsigned int* _nursePreferences; // Array of nurse preferences
unsigned int* _sdscFreeDays; // Number of demanded free days in seven day sequence
unsigned int* _erRoster; // Extended roster - one day for every nurse
unsigned int* _erShiftMap; // Extended shift map
unsigned int* _staffGapsInitOriginal;
unsigned int* _staffGapsInit;
unsigned int* _staffGaps;
unsigned int _TESTTotalfeasible = 0;
unsigned int _TESTTotalruns = 0;
unsigned int _TESTSolved = 0;
unsigned int _TESTLocalFound = 0;
unsigned int _TESTTotalSolvedRun = 0;
float _TESTTotalfirst = 0;
float _TESTTotalsecond = 0;
float _TESTTotalQuality = 0;
float _BPartTime = 0;
#ifdef WATCHDOG
	struct timeb _startTime; // The time, when algorithm started
	//double _stopTime[] = {162,188,281,355,415,373,614,337,403,367,473,609,430,606,912,586,692,940,1075,857,1506,1622,892,830,1039,1138,1152,2401,1519,1963,1212,1300}; // Difference between start and stop time
	double _stopTime = 9999999;
	unsigned int _inputFileNumb = 0;
#endif
#ifdef LOCAL_SEARCH
	unsigned int* _bestShiftList;
#endif

//Initialize structures
void initialization()
{
	cudaDeviceProp  prop;
    int dev;
	time_t seed;
	//_shiftMap = malloc(sizeof(int)*(1 + (_nurseCount*_dayCount-1)/32));
	//_shiftMap = (unsigned int*)malloc(sizeof(int)*(_nurseCount)*_paralelInstanceCount);
	cudaHostAlloc((void**)&_shiftMap, sizeof(int)*(_nurseCount)*_paralelInstanceCount, cudaHostAllocDefault);

	//_currentRoster = (unsigned int*)malloc(sizeof(int)*(2*_nurseCount)*_paralelInstanceCount);
	cudaHostAlloc((void**)&_currentRoster, sizeof(int)*(2*_nurseCount)*_paralelInstanceCount, cudaHostAllocDefault);

	//_bestRoster = malloc(sizeof(int)*(1+(_dayCount*_nurseCount-1)/16));
	//_bestRoster = (unsigned int*)malloc(sizeof(int)*(2*_nurseCount));
	_bestRoster = (unsigned int*)calloc(sizeof(int),(2*_nurseCount));

	//_penaltyFunctions = (unsigned int*)malloc(sizeof(int)*_nurseCount*_paralelInstanceCount);
	cudaHostAlloc((void**)&_penaltyFunctions, sizeof(int)*_nurseCount*_paralelInstanceCount, cudaHostAllocDefault);

	_penaltyFunctionsTest = (unsigned int*)malloc(sizeof(int)*_nurseCount*_paralelInstanceCount);
	_nonAssignedShifts = (unsigned int*)malloc(sizeof(int)*_nurseCount*_paralelInstanceCount);
#ifdef WATCHDOG
	ftime(&_startTime);
#endif

	_bestUtility = INTMAX;
	seed = time(NULL);
	srand((unsigned int)seed);

	_shiftToReroster = _nurseCount*_dayCount;
	_shiftList = (unsigned int*)malloc(sizeof(int)*_shiftToReroster*_paralelInstanceCount);
	_staffGaps = (unsigned int*)calloc(3*_dayCount*_paralelInstanceCount, sizeof(unsigned int));


#ifdef LOCAL_SEARCH
	_bestShiftList = (unsigned int*)malloc(sizeof(int)*_shiftToReroster);
#endif

	cudaGetDevice( &dev );
	printf( "ID of current CUDA device: %d\n", dev );
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 2;
	prop.minor = 0;
	cudaChooseDevice( &dev, &prop );
	printf( "ID of current CUDA device: %d\n", dev );
	cudaSetDevice( dev );
}

//Free all alocated structures
void cleaner()
{
	free(_bestShiftList);
	free(_bestRoster);
	//free(_currentRoster);
	cudaFreeHost(_currentRoster);
	free(_nonAssignedShifts);
	if(_settings.inputDataSetType == MAENHOUT)
		free(_originalRoster);
	free(_penaltyFunctions);
	cudaFreeHost(_penaltyFunctions);
	//free(_penaltyFunctionsTest);
	//free(_shiftMap);
	cudaFreeHost(_shiftMap);
	free(_shiftMapInit);
	free(_shiftList);
	free(_staffGaps);
	//free(_nursePreferences);
	free(_currentRosterInit);
}





//Method for rerostering - CPU version of GPU algorithm
void kerlen_run(unsigned int* shiftList,unsigned int* shiftMap, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* nonAssignedShifts, unsigned int* staffGaps)
{
	unsigned int i = 0;
	unsigned int j = 0;
	unsigned int numInstance = 0;
	unsigned int* index; // = (unsigned int*)calloc(_paralelInstanceCount, sizeof(int)); //index of actual rerostered shift ins list of shifts
	unsigned int* changesCount = (unsigned int*)calloc(_paralelInstanceCount, sizeof(int)); // number of already made changes
	unsigned int utility = 0;
	unsigned int* firstPhase; // = (unsigned int*)malloc(_paralelInstanceCount*sizeof(int));
	unsigned int* infeasible = (unsigned int*)calloc(_paralelInstanceCount, sizeof(int));
	unsigned int* actualShifts = (unsigned int*)malloc(sizeof(int)*_nurseCount*_paralelInstanceCount);
	unsigned int* shiftSecondPhase = (unsigned int*)malloc(_paralelInstanceCount*sizeof(int));
	unsigned int* saveBestFlag = (unsigned int*)calloc(_paralelInstanceCount, sizeof(int));
	unsigned int run = 0;
	unsigned int localSearch = 0;
	unsigned int runsWithoutSucess = 0;
	float probability = 1;
	unsigned int runsWithSameProbability = 0;
	unsigned int offset = 0;
	struct timeb currentTime;
	struct timeb endTime;

	cudaHostAlloc((void**)&index, _paralelInstanceCount* sizeof(int), cudaHostAllocDefault);
	for(i = 0; i < _paralelInstanceCount; i++)
		index[i] = 0;
	cudaHostAlloc((void**)&firstPhase, _paralelInstanceCount*sizeof(int), cudaHostAllocDefault);

	for(i = 0; i < _paralelInstanceCount; i++)
		firstPhase[i] = 1;

	for(i = 0; i < _shiftToReroster; i++)
	{
		//shiftList[_nurseCount*(i%_dayCount)+(i/_dayCount)]=i;
		shiftList[i] = i;
	}

	for(i = 0; i < _nurseCount; i++)
	{
		for(j = 0; j < _dayCount; j++)
		{
			if((getOrigRosterShift(shiftList[i*_dayCount+j]) == FREE)||((getBitOfBitArray(_shiftMapInit, shiftList[i*_dayCount+j]))&&(!getBitOfBitArray(_absenceArray, shiftList[i*_dayCount+j]))))
			{
				int k = (i+1)*_dayCount-1;
				int tmp;
				while(((getOrigRosterShift(shiftList[k])==FREE)||((getBitOfBitArray(_shiftMapInit, shiftList[k]))&&(!getBitOfBitArray(_absenceArray, shiftList[k]))))&&(!(k==i*_dayCount+j)))
					k--;
				if(((k%_dayCount) < _nonFreeDays)||(k==i*_dayCount+j))
					break;
				tmp = shiftList[k];
				shiftList[k] = shiftList[i*_dayCount+j];
				shiftList[i*_dayCount+j] = tmp;
			}
		}
	}

	for(i = 1; i <_paralelInstanceCount; i++) // Copy shift list of first instance to others instances
	{
		for(j = 0; j < _shiftToReroster; j++)
			shiftList[i*_shiftToReroster + j] = shiftList[j];
	}

	for(i = 0; i < _paralelInstanceCount; i++)
		index[i] = _dayCount;

	while(1)
	{
		for(numInstance = 0; numInstance < _paralelInstanceCount; numInstance++)
		{
			offset = _nurseCount*numInstance;
			if((index[numInstance] == _dayCount)||infeasible[numInstance]||changesCount[numInstance]>_shiftChangeMaximum)
			{
				ftime(&currentTime);
				//if((run >= _runCount)||(difftime(currentTime.time, _startTime.time)>_stopTime[_inputFileNumb]))
				if((run >= _runCount)||(difftime(currentTime.time, _startTime.time)>_stopTime))
				{
					_TESTTotalSolvedRun++;
					printf("Time: %f\n", difftime(currentTime.time, _startTime.time)+(currentTime.millitm-_startTime.millitm)/(float)1000);
					goto finish;
					//return;
				}
				saveBestFlag[numInstance] = 0;
				infeasible[numInstance] = 0;
				changesCount[numInstance] = 0;
				for(i = 0; i < _nurseCount; i++)
					nurseInitialize(currentRoster+2*offset, shiftMap+offset, shiftList+_shiftToReroster*numInstance, nonAssignedShifts+offset, staffGaps+_dayCount*3*numInstance, localSearch, probability, i);
				index[numInstance] = 0;
				firstPhase[numInstance] = 1;
			}
			if(firstPhase[numInstance])
			{
				_TESTTotalfirst++;
				for(i = 0; i < _nurseCount; i++)
					nurseLoadFirstPhase(index[numInstance], shiftList + _shiftToReroster*numInstance, actualShifts + offset, i);
			}
			else
			{
				_TESTTotalsecond++;
				changesCount[numInstance]++;
				for(i = 0; i < _nurseCount; i++)
					nurseLoadSecondPhase(shiftSecondPhase[numInstance], actualShifts + offset, i);
			}
		}
		//GPU section

		//printf("\n %d **** \n", run);
		//printRoster(currentRoster);
		//getchar();
		ftime(&currentTime);
		#ifdef DEVICE
			deviceCompatibilityPenalizationHost(actualShifts , _dayCount, shiftMap, firstPhase, currentRoster, penalityFunctions);
			//Code for correctness checking
			/*for(numInstance = 0; numInstance < _paralelInstanceCount; numInstance++)
			{
				offset = _nurseCount*numInstance;
				for(i = 0; i < _nurseCount; i++)
				{
					nurseCompatibilityPenalization(_dayCount, actualShifts + offset, currentRoster + 2*offset, _penaltyFunctionsTest + offset, shiftMap + offset, firstPhase[numInstance], i);
				}
			}
			for(numInstance = 0; numInstance < _paralelInstanceCount*_nurseCount; numInstance++)
			{
				if(penalityFunctions[numInstance] != _penaltyFunctionsTest[numInstance])
				{
					printf("Error - i:%d, n:%d, s:%d CPU:%d GPU:%d\n", numInstance/_nurseCount, numInstance%_nurseCount, actualShifts[numInstance], _penaltyFunctionsTest[numInstance], penalityFunctions[numInstance]);
				}
			}*/
		#else
		for(numInstance = 0; numInstance < _paralelInstanceCount; numInstance++)
		{
			offset = _nurseCount*numInstance;
			for(i = 0; i < _nurseCount; i++)
			{
				nurseCompatibilityPenalization(_dayCount, actualShifts + offset, currentRoster + 2*offset, penalityFunctions + offset, shiftMap + offset, firstPhase[numInstance], i);
			}
		}
		#endif
		ftime(&endTime);
		_BPartTime += endTime.time - currentTime.time + (endTime.millitm - currentTime.millitm)/(float)1000;
		//for(numInstance = 0; numInstance < _paralelInstanceCount*_nurseCount; numInstance++)
		//{
		//	penalityFunctions[numInstance] = 0;
		//}

		for(numInstance = 0; numInstance < _paralelInstanceCount; numInstance++)
		{
			unsigned int min = INTMAX, minCount = 0;
			int minIndex = -1;

			offset = _nurseCount*numInstance;
			if(firstPhase[numInstance])
			{
				for(i = 0; i < _nurseCount; i++)
					nurseAssign(actualShifts + offset, penalityFunctions + offset, currentRoster + 2*offset, nonAssignedShifts + offset, shiftMap + offset, staffGaps + 3*_dayCount*numInstance, i);
			}

			//printRoster(currentRoster);
			//printf("\n");
			//getchar();

			if(!firstPhase[numInstance])
			{
				for(i = 0; i < _nurseCount; i++)
				{
					if(((penalityFunctions + offset)[i] < min))// &&(getOrigRosterShift(i*_dayCount + shiftSecondPhase%_dayCount)!=getOrigRosterShift(shiftSecondPhase)))
					{
						min = (penalityFunctions + offset)[i];
						minIndex = i;
						minCount = 1;
					}
					else if((min != INTMAX)&&((penalityFunctions + offset)[i] == min))
					{
						minCount++;
						if((rand()/(double)(RAND_MAX+1.0)) < 1/((double)minCount))
						{
							minIndex = i;
						}
					}
				}
			}
			if(!firstPhase[numInstance])
			{
				if(minIndex > -1)
				{
					//min = minIndex;
					setRosterShift(currentRoster + 2*offset, minIndex*_dayCount + (actualShifts+offset)[minIndex]%_dayCount, getOrigRosterShift(shiftSecondPhase[numInstance]));
					setBitOfBitArray(shiftMap + offset, minIndex*_dayCount + (actualShifts + offset)[minIndex]%_dayCount, 1);
				}
				else
				{
					infeasible[numInstance] = 1;
					/*printf("Infeasible %d - %d \n", getRow(shiftSecondPhase[numInstance], _dayCount), getColumn(shiftSecondPhase[numInstance], _dayCount));
					for(j = 0; j < _nurseCount; j++)
						printf("%d ", (penalityFunctions + offset)[j]);
					printf("\n");*/
					run++;
					_TESTTotalruns++;
				}
			}

			minIndex = -1;
			for(i = 0; i < _nurseCount; i++)
			{
				if((nonAssignedShifts+offset)[i] != INTMAX)
				{
					minIndex = i;
					break;
				}
			}
			if(minIndex == -1)
			{
				firstPhase[numInstance] = 1;
				index[numInstance]++;
			}
			else
			{
				firstPhase[numInstance] = 0;
				shiftSecondPhase[numInstance] = (nonAssignedShifts+offset)[minIndex];
				(nonAssignedShifts+offset)[minIndex] = INTMAX;
			}
			if((index[numInstance] == _dayCount)&&(!infeasible[numInstance]))
			{
				_TESTTotalfeasible++;
				utility = utilityFunction(currentRoster+2*offset);
				run++;
				_TESTTotalruns++;
				if(_bestUtility > utility)
				{
					saveBestFlag[numInstance] = 1;
					_bestUtility = utility;
					runsWithoutSucess = 0;
					if(localSearch)
						_TESTLocalFound++;
				}
			}

			if(saveBestFlag[numInstance])
			{
				for(i = 0; i < _nurseCount; i++)
					nurseSaveBest(currentRoster+2*offset, shiftList + _shiftToReroster*numInstance, i);
			}
			else
			{
				if((index[numInstance] == _dayCount)&&(!infeasible[numInstance]))
				{
					if(!localSearch)
					{
						if(_bestUtility != INTMAX)
							runsWithoutSucess++;
						if(runsWithoutSucess > LOCAL_SEARCH_RWS_TRESHOLD)
						{
							localSearch=1;
							probability = 0.25;
							runsWithoutSucess = 0;
						}
					}
					else
					{
						runsWithSameProbability++;
						if(runsWithSameProbability > LOCAL_SEARCH_RSP_TRESHOLD)
						{
							probability /= 2;
						}
						if(probability < 0.001)
						{
							localSearch = 0;
							runsWithSameProbability = 0;
						}
					}
				}
			}
		}
	}
	finish:
	if(_bestUtility != INTMAX)
	{
		_TESTTotalSolvedRun += run-1;
	}
	//free(index);
	cudaFreeHost(index);
	free(changesCount);
	//free(firstPhase);
	cudaFreeHost(firstPhase);
	free(infeasible);
	free(actualShifts);
	free(shiftSecondPhase);
	free(saveBestFlag);
}


//Method for nurse to save actual roster row to storage for best one
void nurseSaveBest(unsigned int* currentRoster, unsigned int* shiftList, int nurse)
{
	unsigned int i;
	_bestRoster[2*nurse] = currentRoster[2*nurse];
	_bestRoster[2*nurse+1] = currentRoster[2*nurse+1];
	for(i = 0; i < _dayCount; i++)
		_bestShiftList[nurse*_dayCount + i] = shiftList[nurse*_dayCount + i];
}

//Method for nurse to assign
void nurseAssign(unsigned int* actualShifts, unsigned int* penalityFunction, unsigned int* currentRoster, unsigned int* nonAssignedShifts, unsigned int* shiftMap, unsigned int* staffGaps, int nurse)
{
	if((!getBitOfBitArray(_shiftMapInit, actualShifts[nurse]))||(getBitOfBitArray(_absenceArray, actualShifts[nurse])))
	{
		if(getOrigRosterShift(actualShifts[nurse])!=FREE)
		{
			if(penalityFunction[nurse] != INTMAX)
			{
				setRosterShift(currentRoster, actualShifts[nurse], getOrigRosterShift(actualShifts[nurse]));
				setBitOfBitArray(shiftMap, actualShifts[nurse], 1);
			}
			else
			{
				if(staffGaps[3*getColumn(actualShifts[nurse], _dayCount)+getOrigRosterShift(actualShifts[nurse])-1] > 0)
					staffGaps[3*getColumn(actualShifts[nurse], _dayCount)+getOrigRosterShift(actualShifts[nurse])-1]-=1;
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

void nurseCompatibilityPenalization(int size, unsigned int* actualShifts, unsigned int* currentRoster, unsigned int* penalityFunctions, unsigned int* shiftMap, unsigned int isFirst, int nurse)
{
	penalityFunctions[nurse] = CompatibilityPenalization(nurse*_dayCount+actualShifts[nurse]%_dayCount, size, getOrigRosterShift(actualShifts[nurse]), shiftMap, isFirst, currentRoster);
}

void nurseLoadSecondPhase(unsigned int shiftSecondPhase, unsigned int* actualShifts, int nurse)
{
	actualShifts[nurse] = shiftSecondPhase;
}

void nurseLoadFirstPhase(int index, unsigned int* shiftList, unsigned int* actualShifts, int nurse)
{
	actualShifts[nurse] = shiftList[nurse*_dayCount+index];
}

void nurseInitialize(unsigned int* currentRoster, unsigned int* shiftMap, unsigned int* shiftList, unsigned int* nonAssignedShift, unsigned int* staffGaps, int isLocal, float probability, int nurse)
{
	unsigned int i;
	int swap, swapIndex;
	nonAssignedShift[nurse] = INTMAX;
	currentRoster[2*nurse] = _currentRosterInit[2*nurse];
	currentRoster[2*nurse+1] = _currentRosterInit[2*nurse+1];
	shiftMap[nurse] = _shiftMapInit[nurse];
	for(i = nurse; i < _dayCount; i+=_nurseCount)
	{
		staffGaps[i*3] = _staffGapsInit[i*3];
		staffGaps[i*3+1] = _staffGapsInit[i*3+1];
		staffGaps[i*3+2] = _staffGapsInit[i*3+2];
	}
	/*if(isLocal)
	{
		for(i = 0; i < _dayCount; i++)
		{
			shiftList[nurse*_dayCount + i] = _bestShiftList[nurse*_dayCount + i];
		}
	}*/
	for(i = 0; i < _nonFreeDays; i++)
	{
		//if((!isLocal)||((rand()/(double)(RAND_MAX+1.0)) < probability))
		//{
			swapIndex = (unsigned int)((rand()/(double)(RAND_MAX+1.0))*_nonFreeDays);
		//}
		//else
		//{
		//	swapIndex = i;
		//}
		swapIndex = nurse*_dayCount+swapIndex;
		swap = shiftList[swapIndex];
		shiftList[swapIndex] = shiftList[nurse*_dayCount + i];
		shiftList[nurse*_dayCount + i] = swap;
	}
}

int main(int argc, char *argv[])
{
	int fileIndex = 1; // index of file to proceed
	printf("settings:");
//#ifdef WATCHDOG
//	printf(" WATCHDOG(%ds)", (int)_stopTime);
//#endif
#ifdef INVERSE_SEARCH
	printf(" INVERSE_SEARCH");
#endif
#ifdef LOCAL_SEARCH
	printf( " LOCAL_SEARCH(%d, %d)", LOCAL_SEARCH_RWS_TRESHOLD, LOCAL_SEARCH_RSP_TRESHOLD);
#endif
	printf("\n***************************\n");

	while(((fileIndex + 1 < argc)||argc == 1) && (parseInput(argc, argv, fileIndex)))
	{
		initialization();
		initializeDevice();
		//run(_shiftList, _shiftMap, _currentRoster);
		//kerlen_run(_shiftList, _shiftMap, _currentRoster, _penaltyFunctions, _nonAssignedShifts, _staffGaps);
		homogenousKernelCall();
		printSolution();
		//printRoster(_originalRoster);
		//printf("-----------------------------------\n");
		//printRoster(_bestRoster);
		freeDevice();
		cleaner();
		fileIndex++;
		//printf("B-part time:%f \n", _BPartTime);
		if((argc == 1)||((_settings.inputDataSetType == MOZPATO)&&(fileIndex + 5 > argc)))
			break;
		printf("\n***************************\n");
#ifdef WATCHDOG
		_inputFileNumb++;
		_BPartTime = 0;
#endif
	}
	//printStatistics((_settings.inputDataSetType==MAENHOUT)?((argc-2)>1?argc-2:1):((argc-5>1?argc-4:1)));
#ifdef _DEBUG
	getchar();
#endif
	cudaDeviceReset();
	if(_settings.inputDataSetType == MOZPATO)
	{
		free(_originalRoster);
		//free(_settings.sdscFreeDays);
	}
	return EXIT_SUCCESS;

}