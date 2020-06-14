

#include <iostream>
#include <string>

#include "Rectify.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Incorrect arguments! : [Input File] [Output File]" << std::endl;
        return -1;
    }

	srand(time(NULL));

	string inputPath = argv[1];

	string exportPath = argv[2];

	int debug = argc >= 4 ? atoi(argv[3]) : false;

	RectPano* rectify = new RectPano(inputPath);

	Mat image = rectify->GetRectImage();
	if (image.data) {
		imwrite(exportPath, image);
	}

	if (debug) {
		//image = msop->GetImageFeature(0);
		//imwrite("Tmp/Feature_0.jpg", image);
		imshow("Display window", image);               // Show our image inside it.
		waitKey(0);
	}

    
    return 0;
}

