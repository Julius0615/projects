#pragma once
#include "DisplayableObject.h"
class DemoBObject :
	public DisplayableObject
{
public:
	DemoBObject(BaseEngine* pEngine);//point to main function
	~DemoBObject(void);
	void Draw(void);
	void DoUpdate(int iCurrentTime);
};

