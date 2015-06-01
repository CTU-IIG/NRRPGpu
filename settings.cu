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

Settings _settings;

void setMaenhout()
{
	_settings.changePenalization = 1000;
	_settings.isDayShiftTypeMin = 0;
	_settings.isFairness = 1;
	_settings.fairnessPenalization = 50;
	_settings.isSevenDaySequenceConstrain = 0;
	_settings.isExtendedRoster = 0;
	_settings.isLeftRightConstrainSoft = 0;
	_settings.lrcsPenalization = 0;
	_settings.isMinMaxHardConstrain = 1;
	_settings.mmhcMinWorking = 10; 
	_settings.mmhcMaxWorking = 20; 
	_settings.mmhcMinConsecutive = 2; 
	_settings.mmhcMaxConsecutive = 5;
	_settings.mmhcMinShiftType = 0; 
	_settings.mmhcMaxShiftType = 20; 
	_settings.mmhcMinShiftTypeConsecutive = 1; 
	_settings.mmhcMaxShiftTypeConsecutive = 5;
	_settings.softInfeasiblePenalization = 100000;
	_settings.isNursePreferences = 1;
	_settings.isFreezeShiftSupport = 0;
	_settings.inputDataSetType = MAENHOUT;
	_settings.isMulticommodityFlowConstrins = 0;
	_settings.mfcNightPenalization = 0;
	_settings.mfcPmPenalization = 0;
	_settings.mfcFreePenalization = 0;
	_settings.isShiftListOptimalization = 1;
	_settings.isMPHeuristic = 0;
	_settings.mphPenalization = 0;
}

void setMozPato()
{
	_settings.changePenalization = 1000;
	_settings.isDayShiftTypeMin = 1;
	_settings.isFairness = 0;
	_settings.fairnessPenalization = 0;
	_settings.isSevenDaySequenceConstrain = 1;
	_settings.isExtendedRoster = 1;
	_settings.isLeftRightConstrainSoft = 0;
	_settings.lrcsPenalization = 500;
	_settings.isMinMaxHardConstrain = 0;
	_settings.mmhcMinWorking = 0; 
	_settings.mmhcMaxWorking = (unsigned int)-1; 
	_settings.mmhcMinConsecutive = 0; 
	_settings.mmhcMaxConsecutive = (unsigned int)-1;
	_settings.mmhcMinShiftType = 0; 
	_settings.mmhcMaxShiftType = (unsigned int)-1; 
	_settings.mmhcMinShiftTypeConsecutive = 0; 
	_settings.mmhcMaxShiftTypeConsecutive = (unsigned int)-1;
	_settings.softInfeasiblePenalization = 0;
	_settings.isNursePreferences = 0;
	_settings.isFreezeShiftSupport = 1;
	_settings.inputDataSetType = MOZPATO;
	_settings.isMulticommodityFlowConstrins = 0;
	_settings.mfcNightPenalization = 0;
	_settings.mfcPmPenalization = 0;
	_settings.mfcFreePenalization = 0;
	_settings.isShiftListOptimalization = 0;
	_settings.isMPHeuristic = 2;
	_settings.mphPenalization = 30;
}