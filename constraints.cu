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

// Method tests if there is hard constrain broken between actual shift and shift before
__inline int isLeftHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster)
{
	int left;
	if(getColumn(index, size) == 0)
	{
		if(_settings.isExtendedRoster)
			left = getExtendedRosterShift(getRow(index, size), _erRoster);
		else
			return 0;
	}
	else
	{
		left = getRosterShift(index-1, currentRoster);
	}
	if(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT)))
		return 1;
	return 0;
}

// Method tests if there is hard constrain broken between actual and next shift
__inline int isRightHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* currentRoster)
{
	int right;
	if(getColumn(index, size) == size-1)
		return 0;

	right = getRosterShift(index+1, currentRoster);
	if((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))
		return 1;
	return 0;
}

// Method tests if there is hard constrain broken for actual shift
int isHardConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst, unsigned int* currentRoster)
{
	if(isFirst || !_settings.isLeftRightConstrainSoft)
	{
		if(isLeftHardConstrainBroken(index, size, shiftType, currentRoster)
				|| isRightHardConstrainBroken(index, size, shiftType, currentRoster))
			return 1;
	}
	if(_settings.isSevenDaySequenceConstrain)
	{
		unsigned int day, shifts, shift, frees, nurse, shiftMapNurse;
		int tmp = 0;
		int i, startDay;
		day = getColumn(index, size);
		nurse = getRow(index, size);
		shiftMapNurse = shiftMap[nurse];
		shiftMapNurse = shiftMapNurse&(~_freePreferenceMap[nurse]);
		shifts = 0;
		frees = 0;
		#ifdef SEVENHARD
		startDay = ((unsigned int)day/7)*7;
		for(i = startDay; i < startDay+7; i++)
		{
			if(i >= (int)_dayCount)
			{
				frees++;
			}
			else
			{
				if(i != day)
				{
					if(i < 0)
						frees += (getBitOfBitArray(_erShiftMap, nurse*_dayCount-i))?0:1;
					else
						frees += (getBitOfBitArray(&shiftMapNurse, i))?0:1;
				}
				else
					frees += (shiftType==FREE)?1:0;
			}
		}
		if(frees < _sdscFreeDays[nurse])
		{
			return 1;
		}
#else
		if(_settings.isExtendedRoster)
			startDay = day-6;	
		else
			startDay = day>5?day-6:0;
		for(i = startDay; i < startDay+7; i++)
		{
			if(i >= (int)_dayCount)
			{
				frees++;
				shifts = shifts<<1;
			}
			else
			{
				if(i != day)
				{
					if(i < 0)
						shift = (getBitOfBitArray(_erShiftMap, nurse*_dayCount-i))?1:0;
					else
						shift = (getBitOfBitArray(&shiftMapNurse, i))?1:0;
				}
				else
					shift = (shiftType==FREE)?0:1;
				shifts += shift;
				shifts = shifts << 1;
				if(!shift)
					frees++;
			}
		}
		if(frees < _sdscFreeDays[nurse])
		{
			return 1;
		}
		for(i = startDay+7; i<(int)day+7; i++) // Check all other seven-day sequencies
		{
			if(i < (int)_dayCount)
				shift = (getBitOfBitArray(&shiftMapNurse, i))?1:0;
			else
				return 0;
			if(!shift) frees++;
			if(!((shifts>>(7-tmp))&1))
				frees--;
			if(frees < _sdscFreeDays[nurse])
				return 1;
			tmp++;
		}
#endif
	}
	if(_settings.isMinMaxHardConstrain)
	{
		unsigned int nurse, countWorking, countConsecutive, countAM, countNIGHT, countNight, countShiftTypeCons, i;
		int actualShift, lastShift;
		countConsecutive = countAM = countNIGHT = countNight = countShiftTypeCons = countWorking = 0;
		lastShift = FREE;
		nurse = getRow(index, size);
		for(i = nurse*size; i < (nurse*size+size); i++)
		{
			if(i == index)
				actualShift = shiftType;
			else
				actualShift = getRosterShift(i, currentRoster);
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
			if((countNIGHT > _settings.mmhcMaxShiftType)||(countAM > _settings.mmhcMaxShiftType)||(countNight > _settings.mmhcMaxShiftType)\
				||(countWorking > _settings.mmhcMaxWorking)||(countConsecutive > _settings.mmhcMaxConsecutive))
				return 1;
		}
	}
	return 0;


	//Old version
	/*
	startDay = day>5?index-6:index-day;
	shifts = 0;
	frees = 0;
	for(i = startDay; i<startDay+7; i++) //Sum number of free days in first seven-day segment
	{
		if(getRow(i, size) != getRow(index, size))
		{
			frees++;
			shifts = shifts<<2;
		}
		else
		{
			if(i != index)
				shift = getRosterShift(i, currentRoster);
			else
				shift = shiftType;
			shifts += shift;
			shifts = shifts<<2;
			if(shift == FREE)
				frees++;
		}
	}
	if(frees < 1)
		return 1;
	day = 0;
	for(i = startDay+7; i<index+7; i++) // Check all other seven-day sequencies
	{
		if(getRow(i,size) == getRow(index, size))
			shift = getRosterShift(i, currentRoster);
		else
			return 0;
		if(shift == FREE) frees++;
		if(((shifts>>(14-2*day))&3) == FREE)
			frees--;
		if(frees < 1)
			return 1;
		day++;
	}
	return 0;
	*/
}
/*
// Method returns value of penalty of utility function between actual shift and shift before
int isLeftSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	if(getColumn(index, size) == 0)
		return 0;

	int left = getRosterShift(index-1, shiftMap, shiftChangeArray);
	if((left == PM)&&(shiftType == PM))
		return 5;
	//...
	return 0;
}

// Method returns value of penalty of utility function between actual and next shift
int isRightSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	if(getColumn(index, size) == size-1)
		return 0;

	int right = getRosterShift(index+1, shiftMap, shiftChangeArray);
	if((right == PM)&&(shiftType == PM))
		return 5;
	//...
	return 0;
}


int isMaximallyOneSideSoftContrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	int nurse = getColumn(index, size);
	if((nurse == 0)||(nurse == size -1))
		return 1;

	int left = getRosterShift(index-1, shiftMap, shiftChangeArray);
	int right = getRosterShift(index+1, shiftMap, shiftChangeArray);

	if((shiftType == left)&&(shiftType == right))
	{
		if((shiftType == PM)||(shiftType == FREE))
			return 0; // Noèní za sebou nebo 3 volné za sebou
		if(shiftType == NIGHT)
		{
			if((nurse > 1)&&(getRosterShift(index-2, shiftMap, shiftChangeArray)==NIGHT))
				return 0; //Víc než 3 ranní zasebou zleva
			if((nurse < size-2)&&(getRosterShift(index+2, shiftMap, shiftChangeArray)==NIGHT))
				return 0; //Víc než 3 volné za sebou zprava
		}
	}

	return 1;
}

int isNoOneSideSoftConstrainBroken(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	int nurse = getColumn(index, size);
		if((nurse == 0)||(nurse == size -1))
			return 1;

		int left = getRosterShift(index-1, shiftMap, shiftChangeArray);
		int right = getRosterShift(index+1, shiftMap, shiftChangeArray);

		if((shiftType == left)&&(shiftType == right))
		{
			if(shiftType == PM)
				return 0; // Noèní za sebou nebo 3 volné za sebou
			if(shiftType == FREE)
			{
				if((nurse > 1)&&(getRosterShift(index-2, shiftMap, shiftChangeArray)==FREE))
					return 0; //Víc než 3 ranní zasebou zleva
				if((nurse < size-2)&&(getRosterShift(index+2, shiftMap, shiftChangeArray)==FREE))
					return 0; //Víc než 3 volné za sebou zprava
			}
			if(shiftType == NIGHT)
			{
				if((nurse > 1)&&(getRosterShift(index-2, shiftMap, shiftChangeArray)==NIGHT))
					return 0; //Víc než 3 ranní zasebou zleva
				if((nurse < size-2)&&(getRosterShift(index+2, shiftMap, shiftChangeArray)==NIGHT))
					return 0; //Víc než 3 volné za sebou zprava
			}
		}

		return 1;
}
*/

// Method computes penalization of utility function for shift rostered in specific position at the current time
unsigned int CompatibilityPenalization(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int isFirst,unsigned int* currentRoster)
{
	unsigned int left, right, conCount = 0, result = 0;
	unsigned int i = 2;

	if(getBitOfBitArray(shiftMap, index))
	{
		return INTMAX;
	}
	if(isHardConstrainBroken(index, size, shiftType, shiftMap, isFirst, currentRoster))
		return INTMAX;
	if(isFirst)
		return 0;

	if(_settings.isMPHeuristic == 1)
	{
		result += 2*_settings.mphPenalization;
		if(getColumn(index, size) == 0)
		{
			if(_settings.isExtendedRoster)
				left = getExtendedRosterShift(getRow(index, size), _erRoster);
			else
				left = FREE;
		}
		else
			left = getRosterShift(index-1, currentRoster);

		if(getColumn(index, size) == size-1)
			right = FREE;
		else
			right = getRosterShift(index+1, currentRoster);
		if((left!=FREE)||(right!=FREE))
			result -= _settings.mphPenalization;
		if((left!=FREE)&&(right!=FREE))
			result -= _settings.mphPenalization;
	}

	if(_settings.isMPHeuristic == 2)
	{
		result += 2*_settings.mphPenalization;
		if(getColumn(index, size) == 0)
		{
			if(_settings.isExtendedRoster)
				left = getExtendedRosterShift(getRow(index, size), _erRoster);
			else
				left = FREE;
		}
		else
			left = getOrigRosterShift(index-1);

		if(getColumn(index, size) == size-1)
			right = FREE;
		else
			right = getOrigRosterShift(index+1);
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))||(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= _settings.mphPenalization;
		if((!(((left == PM)&&((shiftType == NIGHT)||(shiftType == AM)))||((left == AM)&&(shiftType == NIGHT))))&&(!((((right == NIGHT)||(right == AM))&&(shiftType == PM))||((right == NIGHT)&&(shiftType == AM)))))
			result -= _settings.mphPenalization;
	}



	if(_settings.isLeftRightConstrainSoft)
	{
		result+= (isLeftHardConstrainBroken(index, size, shiftType, currentRoster) + isRightHardConstrainBroken(index, size, shiftType, currentRoster))*_settings.lrcsPenalization;
	}
	
	if(_settings.isNursePreferences)
		result += getNursePreference(index, shiftType);
	if(_settings.isMulticommodityFlowConstrins)
	{
		if(shiftType == NIGHT)
			return result;
		if(getColumn(index, size) == 0)
			left = getExtendedRosterShift(getRow(index, size), _erRoster);
		else
			left = getRosterShift(index-1, currentRoster);
		if(getColumn(index,size) == size - 1)
			right = FREE;
		else
			right = getRosterShift(index+1, currentRoster); //!!!!Master error


		if(shiftType == PM)
		{
			conCount = 1;
			if(left == PM)
			{
				while((index >= i)&&(getRosterShift(index-i, currentRoster)==PM)) //!!!!Master error
				{
					if(getRow(index-i, size) == getRow(index, size))
						i++;
					else
						break;
				}
				conCount = i;
			}
			if(right == PM)
			{
				i = 2;
				while(((index + i)<=_nurseCount*_dayCount)&&(getRosterShift(index+i, currentRoster)==PM)) //!!!!Master error
				{
					if(getRow(index-i, size) == getRow(index, size))
						i++;
					else
						break;
				}
				conCount += i - 1;
			}
			if(conCount > 1)
				result += (conCount-1)*_settings.mfcNightPenalization;

		}
		if(shiftType == AM)
		{	
			conCount = 1;
			if(left == AM)
			{
				while((index >= i)&&(getRosterShift(index-i, currentRoster)==AM)) //!!!!Master error
				{
					if(getRow(index-i, size) == getRow(index, size))
						i++;
					else
						break;
				}
				conCount = i;
			}
			if(right == AM)
			{
				i = 2;
				while(((index + i)<=_nurseCount*_dayCount)&&(getRosterShift(index+i, currentRoster)==AM)) //!!!!Master error
				{
					if(getRow(index-i, size) == getRow(index, size))
						i++;
					else
						break;
				}
				conCount += i - 1;
			}
			if(conCount > 3)
				result += (conCount-3)*_settings.mfcPmPenalization;
		}
	}
	return result;
}
/*
int isLeftCompatible(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	//return 0;
	if(getColumn(index, size) == 0)
		return 1;
	if(shiftType == FREE)
		return 1;
	int left = getRosterShift(index-1, shiftMap, shiftChangeArray);
	if(left == shiftType)
		return 0;
	return 1;
}

int isRightCompatible(unsigned int index, int size, unsigned int shiftType, unsigned int* shiftMap, unsigned int* shiftChangeArray)
{
	//return 0;
	if(getColumn(index, size) == size - 1)
		return 1;
	if(shiftType == FREE)
		return 1;
	int right = getRosterShift(index+1, shiftMap, shiftChangeArray);
	if(right == shiftType)
		return 0;
	return 1;
}
*/
// Pøi využití isLeft/RightCompatible se prudce sníží feasibility, protože touto
//heuristikou se snažíme nejprve uložit služby tak, aby vedle sebe nebyli dvì stejné,
//ale tím se nám hned na zaèátku zacpou možná místa pro rozdílné a zbylé služby pak není
//možné rozvrhnout!!!!!
