#include "header.h"
#include "templates.h"
#include "DemoAMain.h"
#include "DemoBObject.h"



DemoAMain::DemoAMain()
	:BaseEngine(6)
{
}


DemoAMain::~DemoAMain()
{
}


void DemoAMain::SetupBackgroundBuffer()
{
	FillBackground(0xff0000);
	for (int iX = 0; iX < GetScreenWidth(); iX++)
		for (int iY = 0; iY < this->GetScreenHeight();iY++)
			switch (rand() % 100){
			case 0: 
				SetBackgroundPixel(iX, iY, 0xFF0000); 
				break;
			case 1:
				SetBackgroundPixel(iX, iY, 0x00FF00);
				break;
			case 2:
				SetBackgroundPixel(iX, iY, 0x0000FF);
				break;
			case 3:
				SetBackgroundPixel(iX, iY, 0xFFFF00);
				break;
			case 4:
				SetBackgroundPixel(iX, iY, 0x00FFFF);
				break;
			case 5:
				SetBackgroundPixel(iX, iY, 0xFF00FF);
				break;
				//default?
		}
}


void DemoAMain::MouseDown(int iButton, int iX, int iY)
{
	printf("%d %d", iX, iY);
	if (iButton == SDL_BUTTON_LEFT){
		DrawRectangle(iX - 10, iY - 10, iX + 10, iY + 10, 0xffff00);
		Redraw(false); 
	}
	else if (iButton == SDL_BUTTON_RIGHT){
		DrawRectangle(iX - 10, iY - 10, iX + 10, iY + 10, 0xff0000);
		Redraw(true); //right click will discard all the previous drawed image
	}

}


void DemoAMain::KeyDown(int iKeyCode)
{
	switch (iKeyCode){
	case SDLK_SPACE: //== ' '
		SetupBackgroundBuffer();
		Redraw(true);
		break;
	}
}


int DemoAMain::InitialiseObjects(void)
{
	//Record the fact that we are changed about the array, 
	//so it does not get used elsewhere without reloading it.
	DrawableObjectsChanged();
	//Destroy any existing objects
	DestroyOldObjects();
	//Create an array to store objects
	//Needs to have room for the NULL at the end.
	CreateObjectArray(2);
	//set the array entry after the last one created to NULL.
	//NULL is used to work out where the end of the array is.
	StoreObjectInArray(0,new DemoBObject(this)); //?? this
	//should use new, since the destroy will delete the previous all. do not use malloc().
	StoreObjectInArray(1, NULL);
	return 0;
}
