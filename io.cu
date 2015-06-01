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

// Print error message
void error(char* message)
{
	printf("!!!Error: %s!!!\n", message);
}

// Parse input file and set structures from input
int parseInput(int argc, char* argv[], int fileIndex)
{
	if(fileIndex == 1)
	{
		if(argv[1][0] == 'p')
		{
			printf("Moz&Pato dataset\n");
			printf("%d instances found\n", max(argc-5, 0));
			setMozPato();
		}
		else if(argv[1][0] == 'm')
		{
			printf("Maenhout dataset\n");
			printf("%d instances found\n", argc-2);
			setMaenhout();
		}
		else
		{
			printf("Syntax error");
		}
	}
	if(_settings.inputDataSetType == MAENHOUT)
	{
		return parseInputMaenhout(argc, argv, fileIndex+1);
	}
	else
	{
		return parseInputMozPato(argc, argv, fileIndex+4);
	}
}

int parseInputMozPato(int argc, char* argv[], int fileIndex)
{
	_initialChangeCount = 0;
	_shiftToReroster = 0;

	if(argc < (fileIndex+1))
	{
		printf("Need some input instances of NRRP");
		return 0;
	}
	else
	{
		FILE* input; 
		unsigned int tmp;
		unsigned int i,j,k;
		char ctmp;

		_shiftChangeMaximum = 100;
		_runCount = 100000;

		printf("File %s\n", argv[fileIndex]);

		if(fileIndex == 5)
		{
			input = fopen(argv[2], "r");
			if(input == NULL)
			{
				error("Input file does not exist.");
				return 0;
			}

			/* ORIGINAL SCHEDULE*/
			_nonFreeDays = 0;
			if(fscanf(input, "%d", &tmp)==EOF) // Read month of year - not used
			{
				error("Original roaster reading error");
				fclose(input);
				exit(0);
			}
			if(fscanf(input, "%d", &_nurseCount)==EOF) // Read number of nurses
			{
				error("Original roaster reading error");
				fclose(input);
				exit(0);
			}
			_dayCount = 28;
			_originalRoster = (unsigned int*)calloc(sizeof(int), _nurseCount*2);
			_nursePreferences = (unsigned int*)calloc(sizeof(int), _nurseCount*_dayCount*4);
			_sdscFreeDays = (unsigned int*)malloc(sizeof(unsigned int)* _nurseCount);
			_erRoster = (unsigned int*)calloc(sizeof(unsigned int), 2);
			_erShiftMap = (unsigned int*)calloc(sizeof(unsigned int), _nurseCount);
			_staffGapsInitOriginal = (unsigned int*)calloc(sizeof(unsigned int), 3*_dayCount);
			for (k = 0; k < _nurseCount; k++)
			{
				i = 0;
				if(fscanf(input, "%d", &tmp) == EOF) // Read nurse ID number - not used
				{
					error("Original roaster reading error");
					fclose(input);
					exit(0);
				}
				for (j = 0; j < _dayCount; j++)
				{
					if(fscanf(input, "  %c", &ctmp) == EOF)
					{
						error("Original roaster reading error");
						fclose(input);
						exit(0);
					}
					if(ctmp != 'F')
					{
						_fairnessAverage++;
						i++;
					}
					switch(ctmp)
					{
					case 'F':
						setOrigRosterShift(k*_dayCount+j, FREE);
						break;
					case 'M':
						setOrigRosterShift(k*_dayCount+j, AM);
						_staffGapsInitOriginal[j*3+1] += 1;
						break;
					case 'T':
						setOrigRosterShift(k*_dayCount+j, PM);
						_staffGapsInitOriginal[j*3+2] += 1;
						break;
					case 'N':
						setOrigRosterShift(k*_dayCount+j, NIGHT);
						_staffGapsInitOriginal[j*3] += 1;
						break;
					default:
						error("Original roaster reading error");
						fclose(input);
						exit(0);
					}
				}
				_nonFreeDays = (i > _nonFreeDays) ? i: _nonFreeDays;
			}

			fclose(input);

			input = fopen(argv[3], "r");
			if(input == NULL)
			{
				error("Input file does not exist.");
				return 0;
			}

			/* NURSE PARNIGHTETERS */
			int number, pattern, penalty, credit, totalShifts, patternType, workload;
			char name[80];
			char lastShifts[7];
			for(i = 0; i < _nurseCount; i++)
			{
				if(fscanf(input, "name:%s number:%d last shifts: %c %c %c %c %c %c %c pattern: %d penalty:%d credit days-off: %d total shifts per period: %d type of pattern: %d n. hours per week:  %d\n", name, &number, &lastShifts[6], &lastShifts[5], &lastShifts[4], &lastShifts[3], &lastShifts[2], &lastShifts[1], &lastShifts[0], &pattern, &penalty, &credit, &totalShifts, &patternType, &workload) == EOF)
				{
					error("Nurse parameters reading error");
					fclose(input);
					exit(0);
				}
				switch(lastShifts[0])
				{
				case 'F':
					setExtendedRosterShift(_erRoster, i, FREE);
					break;
				case 'N':
					setExtendedRosterShift(_erRoster, i, NIGHT);
					break;
				case 'M':
					setExtendedRosterShift(_erRoster, i, AM);
					break;
				case 'T':
					setExtendedRosterShift(_erRoster, i, PM);
					break;
				}
				for(j = 1; j < 7; j++)
				{
					if(lastShifts[j-1] != 'F')
						setBitOfBitArray(_erShiftMap, _dayCount*i+j, 1);
				}
				_sdscFreeDays[i] = (workload == 35)?2:1;
			}
			fclose(input);

			input = fopen(argv[4], "r");
			if(input == NULL)
			{
				error("Input file does not exist.");
				return 0;
			}
			for(i = 0; i < _dayCount; i++)
			{
				fscanf(input, "%d", &tmp);
			}
			for(i = 0; i < 3; i++)
			{
				for(j = 0; j < _dayCount; j++)
				{
					fscanf(input, "%d", &tmp);
					_staffGapsInitOriginal[j*3 + i] -= tmp;
				}
			}
			fclose(input);
		}

		_currentRosterInit = (unsigned int*)calloc(_nurseCount*2, sizeof(int));
		_shiftMapInit = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_absenceArray = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_freePreferenceMap = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_staffGapsInit = (unsigned int*)calloc(sizeof(unsigned int), 3*_dayCount);

		for(i = 0; i < 3*_dayCount; i++)
		{
			_staffGapsInit[i] = _staffGapsInitOriginal[i];
		}

		/*for(i = 0; i < 3; i++)
		{
			for(j = 0; j < _dayCount; j++)
				printf("%d ", staffGaps[3*j + i]);
			printf("\n");
		}*/

		input = fopen(argv[fileIndex], "r");
		if(input == NULL)
		{
			error("Input file does not exist.");
			return 0;
		}
		unsigned int state = 0; //0-nurseID, 1-firstDay, 2-lastDay, 3-nextBlock, 4-nextSister, 5-EOF
		unsigned int nurseChange = 0, firstDay = 0, lastDay;
		char readChar;
		unsigned int firstAbsenceDayIndex = _dayCount;

		while(state != 5)
		{
			switch(state)
			{
			case 0:
				fscanf(input, "%d\n", &nurseChange);
				nurseChange--;
				state = 1;
				break;
			case 1:
				fscanf(input, "%d\n", &firstDay);
				if(_nurseCount == 19)
					firstDay = (firstDay>3)?(firstDay-4):(firstDay+25);
				else
					firstDay = (firstDay>6)?(firstDay-7):(firstDay+24);
				firstAbsenceDayIndex = min(firstAbsenceDayIndex, firstDay);
				state = 2;
				break;
			case 2:
				fscanf(input, "%d\n", &lastDay);
				if(_nurseCount == 19)
					lastDay = (lastDay>3)?(lastDay-4):(lastDay+25);
				else
					lastDay = (lastDay>6)?(lastDay-7):(lastDay+24);
				for(i = firstDay; i <= min(lastDay, _dayCount-1); i++)
				{
					setBitOfBitArray(_shiftMapInit, nurseChange*_dayCount + i, 1);
					tmp = getOrigRosterShift(nurseChange*_dayCount + i);
					if(tmp != FREE)
					{
						_initialChangeCount++;
						//if((!_settings.isDayShiftTypeMin)||(_staffGapsInit[i*3+tmp-1] == 0))
						//{
							setBitOfBitArray(_absenceArray, nurseChange*_dayCount + i, 1);
							setBitOfBitArray(_freePreferenceMap, nurseChange*_dayCount + i, 1);
						//}
						//else
						//{
						//	_staffGapsInit[i*3+tmp-1] -= 1;
						//}
					}
					else
					{
						setBitOfBitArray(_freePreferenceMap, nurseChange*_dayCount + i, 1);
					}
				}
				state = 3;
				break;
			case 3:
				fscanf(input, "%c\n", &readChar);
				if(readChar == 's')
					state = 1;
				else
					state = 4;
				break;
			case 4:
				fscanf(input, "%c\n", &readChar);
				if(readChar == 's')
					state = 0;
				else
					state = 5;
				break;
			}
		}
		fclose(input);
		if(_settings.isFreezeShiftSupport)
		{
			for(i = 0; i < _nurseCount; i++)
			{
				for(j = 0; j < firstAbsenceDayIndex; j++)
				{
					setBitOfBitArray(_shiftMapInit, _dayCount*i+j, 1);
					setRosterShift(_currentRosterInit, _dayCount*i+j, getOrigRosterShift(_dayCount*i+j));
					if(getOrigRosterShift(i*_dayCount+j) == FREE)
						setBitOfBitArray(_freePreferenceMap, i*_dayCount+j, 1);
					_nonFreeDays = min(_dayCount-firstAbsenceDayIndex, _nonFreeDays);
				}
			}
		}
		if(!_settings.isShiftListOptimalization)
			_nonFreeDays = _dayCount;
	}
	printf("Nurses: %d,Days: %d,Absences: %d\n", _nurseCount, _dayCount, _initialChangeCount);
	return 1;
}

int parseInputMaenhout(int argc, char* argv[], int fileIndex)
{
	_initialChangeCount = 0;
	_shiftToReroster = 0;
	if(argc < (fileIndex+1))
	{
		printf("Need some input instances of NRRP");
		return 0;
	}
	else
	{
		FILE* input;
		unsigned int shiftsCount, tmp;
		unsigned int i,j,k;
		int nurseChange, dayChange;

		_shiftChangeMaximum = 100;
		_runCount = 50000;

		printf("File %s\n", argv[fileIndex]);
		//printf("File %s\n", "1101.txt");

		input = fopen(argv[fileIndex], "r");
		//input = fopen("1101.txt", "r");
		if(input == NULL)
		{
			error("Input file does not exist.");
			return 0;
		}

		fscanf(input, "%d\t%d\t%d\t%d\t", &_nurseCount, &_dayCount, &shiftsCount, &_absenceCount);
		printf("Nurses: %d,Days: %d,Absences: %d\n", _nurseCount, _dayCount, _absenceCount);

		for (i = 0; i < _dayCount; i++)
		{
			for (j = 0; j < shiftsCount; j++)
			{
				fscanf(input, "%d", &tmp);
			}
		}

		_originalRoster = (unsigned int*)calloc(sizeof(int), _nurseCount*2);
		_currentRosterInit = (unsigned int*)calloc(sizeof(int), _nurseCount*2);
		_shiftMapInit = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_absenceArray = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_freePreferenceMap = (unsigned int*)calloc(sizeof(int), _nurseCount);
		_nursePreferences = (unsigned int*)malloc(sizeof(int)*_nurseCount*_dayCount*shiftsCount);
		/*cudaHostAlloc((void**)&_freePreferenceMap, sizeof(int)*_nurseCount, cudaHostAllocDefault);
		for(i=0; i<_nurseCount; i++)
			_freePreferenceMap[i] = 0;*/

		/* NURSE PREFERENCES (DENOTING LEAVE DAYS) */
		for (k = 0; k < _nurseCount; k++)
		{	for (j = 0; j < _dayCount; j++)
			{	for(i = 0; i < shiftsCount; i++)
				{
					fscanf(input, "%d", &tmp);
					_nursePreferences[k*_dayCount*shiftsCount+j*shiftsCount+(i+1)%4] = tmp;
					if(_settings.isFreezeShiftSupport && tmp)
					{
						setBitOfBitArray(_shiftMapInit, k*_dayCount+j, 1);
					}
				}
			}
		}

		/* ORIGINAL SCHEDULE*/
		_nonFreeDays = 0;
		for (k = 0; k < _nurseCount; k++)
		{
			i = 0;
			for (j = 0; j < _dayCount; j++)
			{
				fscanf(input, "%d", &tmp);
				if(tmp != 3)
				{
					_fairnessAverage++;
					if(!getBitOfBitArray(_shiftMapInit, k*_dayCount+j))
						i++;
				}
				setOrigRosterShift(k*_dayCount+j,(tmp+1)%4);
				if(getBitOfBitArray(_shiftMapInit, k*_dayCount+j))
				{
					setRosterShift(_currentRosterInit, k*_dayCount+j, (tmp+1)%4);
					if(tmp == 3)
						setBitOfBitArray(_freePreferenceMap, k*_dayCount+j, 1);
				}
			}
			_nonFreeDays = (i > _nonFreeDays) ? i: _nonFreeDays;
		}

		_fairnessAverage /= _nurseCount;

		/* DISRUPTIONS */
		for (i = 0; i < _absenceCount; i++)
		{
			fscanf(input, "%d\t%d\t", &nurseChange, &dayChange);
			setBitOfBitArray(_absenceArray, nurseChange*_dayCount + dayChange, 1);
			_initialChangeCount++;
			setBitOfBitArray(_shiftMapInit, nurseChange*_dayCount + dayChange, 1);
		}

		fclose(input);
	}
	return 1;
}

// Method for printing results (best roster)
void printSolution()
{
	//int i, j;
	if(_bestUtility != INTMAX)
	{
		_TESTSolved++;
		_TESTTotalQuality += (_bestUtility)/(float)(_settings.changePenalization*_initialChangeCount);
		/*for(i = 0; i < _nurseCount; i++)
		{
			for(j = 0; j < _dayCount; j++)
			{
				printf("%d ", getRosterShift(i*_dayCount+j, _bestShiftMap, _bestShiftChangeArray));
			}
			printf("\n");
		}*/
		printf("Utility: %d\n", _bestUtility);
		printf("Quality: %f\n", (_bestUtility)/(float)(1000*_initialChangeCount));
		/*for(i = 0; i < _shiftChangeMaximum; i++)
		{
			if(_bestShiftChangeArray[2*i] != -1)
			{
				if(_bestShiftChangeArray[2*i+1] != -1)
					printf("Change %d: Nurse %d in day %d takes shift of nurse %d\n", i+1, getRow(_bestShiftChangeArray[2*i], _dayCount), getColumn(_bestShiftChangeArray[2*i], _dayCount), getRow(_bestShiftChangeArray[2*i+1], _dayCount));
				else
					printf("Change %d: Nurse %d in day %d has absent\n", i+1, getRow(_bestShiftChangeArray[2*i], _dayCount), getColumn(_bestShiftChangeArray[2*i], _dayCount));
			}
			else
			{
				break;
			}
		}*/
	}
	else
	{
		//printf("No feasible solution found\n");
		printf("Utility: X\n");
	}
}

//Method for printing actual roster
void printRoster(unsigned int* currentRoster)
{
	unsigned int i, j;
	for(i = 0; i < _nurseCount; i++)
	{
		for(j = 0; j < _dayCount; j++)
		{
			printf("%d ", getRosterShift(i*_dayCount+j, currentRoster));
		}
		printf("\n");
	}
}

//Method prints Statistics information - used for testing
void printStatistics(int instances)
{
	printf("Total number of feasible rosters: %d\n",_TESTTotalfeasible);
	printf("Total number of runs: %d\n", _TESTTotalruns);
	printf("Total feasibility: %f\n", _TESTSolved/(float)instances);
	if(_TESTSolved>0)
		printf("Total Quality: %f\n",_TESTTotalQuality/(float)_TESTSolved);
	printf("Average feasibility in runs: %f\n",_TESTTotalfeasible/(float)_TESTTotalruns);
	if(_TESTSolved>0)
		printf("Average feasibility in runs with solution: %f\n",_TESTTotalfeasible/(float)_TESTTotalSolvedRun);
	printf("First: %f\n",_TESTTotalfirst/(_TESTTotalfirst + _TESTTotalsecond));
	printf("Second: %f\n",_TESTTotalsecond/(_TESTTotalfirst + _TESTTotalsecond));
#ifdef LOCAL_SEARCH
	printf("Found while local search: %d\n", _TESTLocalFound);
#endif
}

void printBitArray(unsigned int* bitArray, unsigned int width, unsigned int hight)
{
	unsigned int i, j;
	for(i = 0; i < hight; i++)
	{
		for(j = 0; j < width; j++)
		{
			printf("%d ", getBitOfBitArray(bitArray, _dayCount*i + j)?1:0);
		}
		printf("\n");
	}
}
