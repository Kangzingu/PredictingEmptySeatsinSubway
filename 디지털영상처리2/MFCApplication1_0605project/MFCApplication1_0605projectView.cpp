
// MFCApplication1_0605projectView.cpp : CMFCApplication1_0605projectView 클래스의 구현
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

//
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define CAM_WIDTH 1280 //480
#define CAM_HEIGHT 720//360

/** Function Headers */


#include "opencv/cv.h"
#include "opencv/highgui.h"

#include <iostream>

#define WIN_NAME "얼굴인식"
#define FACE_CLASSIFIER_PATH "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
#define FACE_SEARCH_SCALE 1.1
#define MERGE_DETECTED_GROUP_CNTS 3
#define FACE_FRAME_WIDTH 50
#define FACE_FRAME_HEIGHT 50
#define FACE_FRAME_THICKNESS 5
#define EYE_CLASSIFIER_PATH  "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
//안구 인식 XML 파일 경로 선언
CascadeClassifier eye_cascade;//안구 검출
CascadeClassifier face_cascade;//얼굴 검출

//졸고있음을 인식하는 전역 변수 선언
Mat hand3ss;
int count3ss = 0;
RGBQUAD** aviBuffer1;//이전 버퍼
RGBQUAD** aviBuffer2;//다음 버퍼
int aviHeight;//커넥티드 컴포넌트 라벨링된 결과물의 너비, 높이 정보를 이용
int aviWidth;
int frameCnt = 0;
class ArrValue {
public:
	int x_dif = 0;
	int y_dif = 0;
};



													   // SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
													   // 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFCApplication1_0605project.h"
#endif

#include "MFCApplication1_0605projectDoc.h"
#include "MFCApplication1_0605projectView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


													   // CMFCApplication1_0605projectView

IMPLEMENT_DYNCREATE(CMFCApplication1_0605projectView, CView)

BEGIN_MESSAGE_MAP(CMFCApplication1_0605projectView, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(ID_LOADAVI, &CMFCApplication1_0605projectView::OnLoadavi)
	ON_COMMAND(ID_FACE_DETECT, &CMFCApplication1_0605projectView::OnFaceDetect)
	ON_COMMAND(ID_DS_EYE_DETECT, &CMFCApplication1_0605projectView::OnDsEyeDetect)
	ON_COMMAND(ID_DS_MOTIONCHANGE, &CMFCApplication1_0605projectView::OnDsMotionchange)
	ON_COMMAND(ID_DS_FACEDOWN, &CMFCApplication1_0605projectView::OnDsFacedown)
	ON_COMMAND(ID_DS_EDGEDECTOR, &CMFCApplication1_0605projectView::OnDsEdgedector)
	ON_COMMAND(ID_DS_LABEL, &CMFCApplication1_0605projectView::OnDsLabel)
	ON_COMMAND(ID_DS_ROTATEFACE, &CMFCApplication1_0605projectView::OnDsRotateface)
	ON_COMMAND(ID_DS_HANDDETECTOR32784, &CMFCApplication1_0605projectView::OnDsHanddetector32784)
END_MESSAGE_MAP()

// CMFCApplication1_0605projectView 생성/소멸

CMFCApplication1_0605projectView::CMFCApplication1_0605projectView()
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CMFCApplication1_0605projectView::~CMFCApplication1_0605projectView()
{
}

BOOL CMFCApplication1_0605projectView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// CMFCApplication1_0605projectView 그리기

void CMFCApplication1_0605projectView::OnDraw(CDC* /*pDC*/)
{
	CMFCApplication1_0605projectDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
}


// CMFCApplication1_0605projectView 인쇄

BOOL CMFCApplication1_0605projectView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void CMFCApplication1_0605projectView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CMFCApplication1_0605projectView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}


// CMFCApplication1_0605projectView 진단

#ifdef _DEBUG
void CMFCApplication1_0605projectView::AssertValid() const
{
	CView::AssertValid();
}

void CMFCApplication1_0605projectView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCApplication1_0605projectDoc* CMFCApplication1_0605projectView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCApplication1_0605projectDoc)));
	return (CMFCApplication1_0605projectDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCApplication1_0605projectView 메시지 처리기
void CMFCApplication1_0605projectView::OnLoadavi()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;

	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");

	for (;;) {
		Mat frame;
		Capture >> frame;
		if (frame.data == nullptr) {
			break;
		}
		imshow("video", frame);
		if (waitKey(30) >= 0)break;
	}
	AfxMessageBox("Copleted");

}


//그냥 얼굴에 박스 쳐주는 함수인것임
void CMFCApplication1_0605projectView::OnFaceDetect()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	//메뉴 기능을 누를 때 호출되는 함수

	// 웹 캠 생성
	VideoCapture capture(0);

	// 웹 캠을 실행하지 못한 경우 에러 출력 및 종료
	if (!capture.isOpened()) {
		cerr << "ERROR : 웹 캠 디바이스를 찾을 수 없음" << std::endl;
		//return;//종료합니다.
	}

	// 윈도우 생성
	namedWindow(WIN_NAME, 1);

	// 얼굴인식 템플릿 설정
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);

	Mat frameOriginalMat;
	Mat frame;
	vector<Rect> faces;
	while (true) {
		
		bool isFrameValid = true;
		try {
			// 웹 캠 프레임의 원본 크기 저장
			capture >> frameOriginalMat;

			// 원본 크기의 1/2로 축소 (왜냐면 프레임의 크기가 클 경우 연산시간이 증가)
			resize(frameOriginalMat, frame, cv::Size(480, 320), 0, 0, CV_INTER_NN);
		}
		catch (cv::Exception& e) {
			// 에러 출력
			std::cerr << "프레임 축소에 실패했기에, 해당 프레임을 무시합니다." << e.err << std::endl;
			isFrameValid = false;
		}

		// 프레임 크기 축소에 성공한 경우 얼굴인식
		if (isFrameValid) {
			try {
				// 프레임을 그레이 스케일 및 이퀄라이즈 처리
				Mat grayframe;
				cvtColor(frame, grayframe, CV_BGR2GRAY);
				equalizeHist(grayframe, grayframe);

				// 얼굴인식 템플릿을 이용하여 얼굴인식
				face_classifier.detectMultiScale(
					grayframe, faces,
					FACE_SEARCH_SCALE,
					MERGE_DETECTED_GROUP_CNTS,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
				);

				for (int i = 0; i < faces.size(); i++) {
					
						// 얼굴인식 사각형 틀 출력
						Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
						Point tr(faces[i].x, faces[i].y);
						rectangle(frame, lb, tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);
					
				}

				// 윈도우에 결과 출력
				//cv::imshow(WIN_NAME, frame);//이렇게 하면 영상 프레임 하나가 순수하게 출력되겠지
				cv::imshow(WIN_NAME, frame);////여기 고쳐야돼

				//여기까지 프레임 처리하고
				//여기서 3ss 진행하면 될듯 합니다
				

			}
			catch (cv::Exception& e) {
				cerr << "얼굴인식 처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
			}
			
		}

		int keyCode = cv::waitKey(30);

		// esc 키가 눌리면 프레임 캡쳐 종료
		if (keyCode == 27) {
			break;
		}
	}
	

	return;//종료합니다

}




/**
* 영상에서 얼굴이랑 눈을 추출한다
*
* @param  im    영상
* @param  tpl   Will be filled with the eye template, if detection success.
* @param  rect  Will be filled with the bounding box of the eye
* @return zero=failed, nonzero=success
*/
vector<cv::Rect> faces, eyes;

int detectEye(Mat& im, Mat& tpl, Rect& rect)
{
	
	face_cascade.detectMultiScale(im, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (int i = 0; i < faces.size(); i++)
	{
		Mat face = im(faces[i]);
		eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

		if (eyes.size())
		{
			rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
			tpl = im(rect);
		}
	}

	return eyes.size();

}

/**
* 눈을 찾아줘
*
* @param   im    : 영상
* @param   tpl   : 눈 탬플릿
* @param   rect  : 눈영역을 바운딩 박스로 쳐줌
*/
void trackEye(Mat& im, Mat& tpl, Rect& rect)
{
	Size size(rect.width * 2, rect.height * 2);
	Rect window(rect + size - Point(size.width / 2, size.height / 2));

	window &= Rect(0, 0, im.cols, im.rows);//(0,0)~(이미지의 열, 이미지의 행)

	Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);
	matchTemplate(im(window), tpl, dst, CV_TM_SQDIFF_NORMED);

	double minval, maxval;
	Point minloc, maxloc;
	minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

	if (minval <= 0.2)
	{
		rect.x = window.x + minloc.x;
		rect.y = window.y + minloc.y;
	}
	else
		rect.x = rect.y = rect.width = rect.height = 0;
}

void CMFCApplication1_0605projectView::OnDsEyeDetect()//안구 검출
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// cascade classifiers를 불러옵니다 =>xml 불러오는 디렉토리 주의
	face_cascade.load(FACE_CLASSIFIER_PATH);
	eye_cascade.load(EYE_CLASSIFIER_PATH);

	// 웹캠 설정
	VideoCapture cap(0);

	
	if (face_cascade.empty() || eye_cascade.empty() || !cap.isOpened())
		return ;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);//너비 320
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);//폭 240

	Mat frame, eye_tpl;//영상 프레임/ 눈 템플릿
	Rect eye_bb;//눈 바운딩 박스

	while (waitKey(15) != 'q')//q라는 신호가 들어올때까지 대기함
	{
		cap >> frame;
		if (frame.empty())
			break;

		// Flip the frame horizontally, Windows users might need this
		flip(frame, frame, 1);

		// Convert to grayscale and 
		// adjust the image contrast using histogram equalization
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);

		if (eye_bb.width == 0 && eye_bb.height == 0)
		{
			// Detection stage
			// Try to detect the face and the eye of the user
			detectEye(gray, eye_tpl, eye_bb);
		}
		else
		{
			// Tracking stage with template matching
			trackEye(gray, eye_tpl, eye_bb);

			// Draw bounding rectangle for the eye
			rectangle(frame, eye_bb, CV_RGB(0, 255, 0));
		}

		// Display video
		imshow("video", frame);
	}

	return ;//함수를 종료

}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//얼굴모양이 라벨링 되어 있을 경우, 그 프레임이 움직이는 이동량의 임계치를 계산함

double calculate_subtraction(int x, int y, int x_dif, int y_dif, int n) {
	int sum = 0;
	for (int i = -8; i < 8; i++)
	{
		for (int j = -8; j < 8; j++)
		{
			if (!((x + x_dif + i) <= 0 || (x + x_dif + i) >= aviWidth || (y + y_dif + j) <= 0 || (y + y_dif + j) >= aviHeight))
			{
				if (n % 2 == 1)
				{
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbRed);
				}
				else
				{
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbBlue - (int)aviBuffer2[y + j][x + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbGreen - (int)aviBuffer2[y + j][x + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbRed - (int)aviBuffer2[y + j][x + i].rgbRed);
				}
			}
			else {
				if (n % 2 == 1) {
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed);
				}
				else
				{
					sum += abs((int)aviBuffer2[y + j][x + i].rgbBlue);
					sum += abs((int)aviBuffer2[y + j][x + i].rgbGreen);
					sum += abs((int)aviBuffer2[y + j][x + i].rgbRed);
				}
			}
		}
	}
	return sum / 16.0 / 16;
}

ArrValue On3SS(int x, int y, ArrValue av, int number, int n) {
	int w = (int)pow(2, number); //처음 시작이 3이거든.
	int value = 0;
	double arr[9];
	double min = 800.0;
	int min_idx;
	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			int num = (j + 1) * 3 + (i + 1);
			int x2 = (i*w) + av.x_dif;
			int y2 = (j*w) + av.y_dif;
			//			if (!(x< 0 || x>aviWidth || y <0 || y>aviHeight) && !(x+x2<0 || x+x2>aviWidth|| y+y2<0||y+y2>aviHeight))
			if (!(x + x2 <= 0 || x + x2 >= aviWidth || y + y2 <= 0 || y + y2 >= aviHeight || x <= 0 || x >= aviWidth || y <= 0 || y >= aviHeight))
				arr[num] = calculate_subtraction(x, y, x2, y2, n);
			else
				arr[num] = 800;
		}
	}
	for (int i = 0; i < 9; i++)
	{
		if (arr[i] <= min && arr[i] >= 0)
		{
			min_idx = i;
			min = arr[i];
		}
	}
	av.x_dif += (min_idx % 3 - 1)*w;
	av.y_dif += (min_idx / 3 - 1)*w;
	if (number > 1)
	{
		number--;
		av = On3SS(x, y, av, number, n);
	}
	return av;
}


void On3ss(Mat frame) {//frame을 받아서 3ss를 진행할거라구
	//// TODO: 여기에 명령 처리기 코드를 추가합니다.

	////frame을 받아서 3ss를 진행

	//if (frame.data == nullptr)
	//	AfxMessageBox("Error Video");

	////크기만큼 생성
	//aviHeight = frame.rows;
	//aviWidth = frame.cols;
	//aviBuffer1 = new RGBQUAD*[aviHeight];
	//aviBuffer2 = new RGBQUAD*[aviHeight];

	//for (int i = 0; i < aviHeight; i++)
	//{
	//	aviBuffer1[i] = new RGBQUAD[aviWidth];
	//	aviBuffer2[i] = new RGBQUAD[aviWidth];
	//}

	////3SS을 위한 준비
	//FILE* file = fopen("handMotion.txt", "wt");
	//CString value3SS;

	////frame 크기의 16분의1을 블록사이즈로 지정
	//int block_w_n = aviWidth / 16;
	//int block_h_n = aviHeight / 16;
	//int value = 0;

	//namedWindow("img1", CV_WINDOW_AUTOSIZE);
	//for (int i = 0; i < frame.size(); i++) {

	//	// 얼굴인식 사각형 틀 출력
	//	Point lb(frame[i].x + faces[i].width, faces[i].y + faces[i].height);
	//	Point tr(faces[i].x, faces[i].y);
	//	rectangle(frame, lb, tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);

	//}
	//imshow("img1", frame);

	//if (count3ss % 2 == 0)
	//{
	//	for (int i = 0; i < aviHeight; i++) {
	//		for (int j = 0; j < aviWidth; j++)
	//		{//BRG순서
	//			aviBuffer2[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];
	//			aviBuffer2[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
	//			aviBuffer2[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
	//			aviBuffer1[i][j].rgbBlue = 0;
	//			aviBuffer1[i][j].rgbGreen = 0;
	//			aviBuffer1[i][j].rgbRed = 0;
	//		}
	//	}
	//}
	//else
	//{
	//	for (int i = 0; i < aviHeight; i++) {
	//		for (int j = 0; j < aviWidth; j++)
	//		{//BRG순서
	//			aviBuffer1[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];
	//			aviBuffer1[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
	//			aviBuffer1[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
	//			aviBuffer2[i][j].rgbBlue = 0;
	//			aviBuffer2[i][j].rgbGreen = 0;
	//			aviBuffer2[i][j].rgbRed = 0;

	//		}
	//	}
	//}

	//if (count3ss >= 1) {//3ss
	//	for (int i = 0; i < block_h_n; i++)
	//	{
	//		for (int j = 0; j < block_w_n; j++)
	//		{
	//			ArrValue av;
	//			av = On3SS(16 * (j + 1) - 8, 16 * (i + 1) - 8, av, 3, cnt);
	//			value3SS.Format(_T("(%3d, %3d) "), av.x_dif, av.y_dif);
	//			fprintf(file, value3SS);
	//		}

	//		fprintf(file, "\n");
	//	}
	//	value3SS.Format(_T("-%d & %d 프레임의 모션변화값- \n\n"), cnt - 1, cnt);
	//	fprintf(file, value3SS);
	//}
	////n이 0일때엔 비교 ㄴㄴ      //n이 1일때부터 3SS시작이고, 
	////n이 짝수일때에는 2->1 을보고      //n이 홀수일때에는 1->2 을봐야지
	//cnt++;
	//////////////////////////
	//fclose(file);


							//AfxMessageBox("Completed");

}

Point getHandCenter(const Mat& mask, double& radius) {
	//거리 변환 행렬을 저장할 변수
	Mat dst;
	distanceTransform(mask, dst, CV_DIST_L2, 5);
	//결과는 CV_32SC1 타입
	//거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.
	int maxIdx[2];//좌표 값을 얻어올 배열(행, 열 순으로 저장됨)
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);//최소값은 사용 X
	return Point(maxIdx[1], maxIdx[0]);

}

void CMFCApplication1_0605projectView::OnDsMotionchange()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	

	///////////////////////////////////////////////////
	Mat resultHandImg, tmpImg, originalHandImg;
	VideoCapture video(0);
	//결과창
	namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("original_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	
	Mat resultHandCenterImg;
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);//경로지정
	
	Mat resultFaceImg;
	vector<Rect> faces;

	//변수 선언
	//크기만큼 생성
	aviHeight = (int)video.get(CAP_PROP_FRAME_HEIGHT);
	aviWidth = (int)video.get(CAP_PROP_FRAME_WIDTH);
	aviBuffer1 = new RGBQUAD*[aviHeight];
	aviBuffer2 = new RGBQUAD*[aviHeight];

	for (int i = 0; i < aviHeight; i++)
	{
		aviBuffer1[i] = new RGBQUAD[aviWidth];
		aviBuffer2[i] = new RGBQUAD[aviWidth];
	}

	//3SS을 위한 기본바탕
	FILE* file = fopen("faceFrame.txt", "wt");
	CString value3SS;
	int block_w_n = aviWidth / 16;
	int block_h_n = aviHeight / 16;
	int value = 0;
	int n = 0;

	//////////////////////////////////////
	while (true) {//반복
		video >> tmpImg;//그냥 정보 임시저장용
		video >> resultHandCenterImg;//손바닥 인식 결과 저장용
		bool isFrameValid = true;
		try {
			// 웹 캠 프레임의 원본 크기 저장
			video >> tmpImg;

			// 원본 크기의 1/2로 축소 할수있징만 나는 하지 않을거야 (왜냐면 프레임의 크기가 클 경우 연산시간이 증가)
			resize(tmpImg, resultFaceImg, Size(tmpImg.cols / 1, tmpImg.rows / 1), 0, 0, CV_INTER_NN);
		}
		catch (Exception& e) {
			// 에러 출력
			cerr << "프레임 축소에 실패했기에, 해당 프레임을 무시합니다." << e.err << std::endl;
			isFrameValid = false;
		}
		if (isFrameValid) {
			try {
				Mat grayframe;
				//그레이 스케일 이미지로 변경
				cvtColor(resultFaceImg, grayframe, CV_BGR2GRAY);
				// 프레임을 그레이 스케일 및 이퀄라이즈 처리
				equalizeHist(grayframe, grayframe);


				// 얼굴인식 템플릿을 이용하여 얼굴인식
				face_classifier.detectMultiScale(
					grayframe, faces,
					FACE_SEARCH_SCALE,
					MERGE_DETECTED_GROUP_CNTS,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
				);

				for (int i = 0; i < faces.size(); i++) {

					// 얼굴인식 사각형 틀 출력
					Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
					Point tr(faces[i].x, faces[i].y);
					rectangle(resultFaceImg, lb, tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);

				}
				// 윈도우에 결과 출력
				//cv::imshow("result_face_image", resultFaceImg);
			}
			catch (cv::Exception& e) {
				cerr << "얼굴인식 처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
			}

		}
		//
		try {
			cvtColor(tmpImg, resultHandImg, CV_BGR2YCrCb);
			//피부 색 범위 설정
			inRange(resultHandImg, Scalar(0, 133, 77), Scalar(255, 173, 127), resultHandImg);

			//8bit 단일채널?
			originalHandImg = (resultHandImg.size(), CV_8UC3, Scalar(0));

			add(tmpImg, Scalar(0), originalHandImg, resultHandImg);

			imshow("result_hand_image", originalHandImg);
			//imshow("original_hand_image", originalHandImg);
			//erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
			double radius;
			for (int i = 0; i < faces.size(); i++) {
				//손 인식 시 얼굴 제외
				Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				Point tr(faces[i].x, faces[i].y);
				for (int j = 0; j < faces[i].height; j++) {
					for (int k = 0; k < faces[i].width; k++) {
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[0] = 0;
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[1] = 0;
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[2] = 0;
					}
				}
			}
			


			//여기에 3ss=>result handImg로 3ss연산함=>움직임을 판단함
			//resulthandimg는 변환된 이미지
			
			if (n % 2 == 0)
			{
				for (int i = 0; i < aviHeight; i++) {
					for (int j = 0; j < aviWidth; j++)
					{//BRG순서
						aviBuffer1[i][j].rgbBlue = resultHandImg.at<Vec3b>(i, j)[0];
						aviBuffer1[i][j].rgbGreen = resultHandImg.at<Vec3b>(i, j)[1];
						aviBuffer1[i][j].rgbRed = resultHandImg.at<Vec3b>(i, j)[2];
					}
				}
			}
			else
			{
				for (int i = 0; i < aviHeight; i++) {
					for (int j = 0; j < aviWidth; j++)
					{//BRG순서
						aviBuffer2[i][j].rgbBlue = resultHandImg.ptr<BYTE>(i, j)[0];
						aviBuffer2[i][j].rgbGreen = resultHandImg.ptr<BYTE>(i, j)[1];
						aviBuffer2[i][j].rgbRed = resultHandImg.ptr<BYTE>(i, j)[2];
					}
				}
			}

			if (n >= 1) {
				for (int i = 0; i <block_h_n; i++)
				{
					for (int j = 0; j < block_w_n; j++)
					{
						ArrValue av;
						av = On3SS(16 * (j + 1) - 8, 16 * (i + 1) - 8, av, 3, n);
						value3SS.Format(_T("(%3d, %3d) "), av.x_dif, av.y_dif);
						fprintf(file, value3SS);
					}

					fprintf(file, "\n");
				}
				value3SS.Format(_T("-%d & %d 프레임의 모션변화값- \n\n"), n - 1, n);//n-1과 n번째 프레임의 모션변화값을 출력합니다=> 텍파로 out
				fprintf(file, value3SS);
			}
			//n이 0일때엔 비교 ㄴㄴ		//n이 1일때부터 3SS시작이고, 
			//n이 짝수일때에는 2->1 을보고		//n이 홀수일때에는 1->2 을봐야지
			n++;


			//손바닥 중심 점 반환
			Point center = getHandCenter(resultHandImg, radius);
			//원으로 손 표시
			circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
			circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
			//imshow("handcenter_image", resultHandCenterImg);
		}
		catch (Exception& e) {
			cerr << "처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}
	//메모리 해제
	video.release();
	tmpImg.release();
	//resultHandImg.release();
	originalHandImg.release();
	destroyAllWindows();

	


}


//////////////////////////////////////////////////////////////////
//////////////////졸고있음을 판단하는 함수 영역//////////////////////
void CMFCApplication1_0605projectView::OnDsFacedown()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	/*
	변수선언
	*/
	int aviHeight;//프레임 높이=>전역으로 선언함
	int aviWidth;//프레임 너비=>전역으로 선언함

	// 웹 캠 생성
	VideoCapture capture(0);

	//웹 캠의 크기만큼 aviHeight와 aviwidth를 설정
	aviHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);//높이는 몇인가
	aviWidth = (int)capture.get(CAP_PROP_FRAME_WIDTH);//너비는 몇인가

	aviBuffer1 = new RGBQUAD*[aviHeight];
	aviBuffer2 = new RGBQUAD*[aviHeight];

	for (int i = 0; i < aviHeight; i++)//객체생성
	{
		aviBuffer1[i] = new RGBQUAD[aviWidth];
		aviBuffer2[i] = new RGBQUAD[aviWidth];
	}

	//3SS을 위한 기본바탕
	FILE* file = fopen("faceFrame.txt", "wt");
	CString value3SS;
	int block_w_n = aviWidth / 16;//블록 폭 사이즈는 일단 프레임크기의 1/16으로 설정함
	int block_h_n = aviHeight / 16;//블록의 높이 사이즈는 프레임크기의 1/16
	int value = 0;//

	int nc = -1;//카운드 값

	///////////////////////////////////////////
	Point pre_lb;//이전 프레임의 left buttom
	Point pre_tr;//이전 프레임의 right top

	int nowTime = 0;//현재 경과하고있는 시간 변수
	int contiTime = 0;//프레임을 잡지 못하였을 때, 연속되게 얼마나 못잡는지를 판단하는 변수
	int errTime = 0;//erro의 flag 가 on된 시점의 시간
	int cnt = 0;//가중치
	///////////////////////////////////////////////////
	

													  
	if (!capture.isOpened()) {// 웹 캠을 실행하지 못한 경우 에러 출력 및 종료
		cerr << "ERROR : 웹 캠 디바이스를 찾을 수 없음" << std::endl;
		//return;//종료합니다.
	}

	// 윈도우 생성
	//namedWindow("snooze", 1);
	namedWindow("original", 2);
	// 얼굴인식 템플릿 설정
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);

	Mat frameOriginalMat;//original frame
	Mat frame;//원본 프레임의 사이즈를 조절한 프레임을 저장

	
	Point lb;
	Point tr;
	vector<Rect> faces;//얼굴이라고 판단되는 영역에 바운딩박스 처리

	//처음 영상
	Mat first_frame;
	vector<Rect> first_face;
	Point first_lb;
	Point first_tr;
	bool isFirst = false;//나 처음이니?

	//마지막 영상
	Mat last_frame;
	vector<Rect> last_face;
	Point last_lb;
	Point last_tr;

	



	while (true) {//waitkey가 특정값이 들어올때까지 loop를 반복함
		nowTime++;//while loop를 한번 돌때마다 시간 값이 하나씩 증가하도록 함
		//nc++;//수행 종료시 카운트값 증가

		bool isFrameValid = true;//프레임이 따와졌는지 확인
		bool frameGone = false;
		try {
			// 웹 캠 프레임의 원본 크기 저장
			capture >> frameOriginalMat;//축소하는과정에서 원본 데이터가 손실할수있기 때문

			// 원본 크기의 1/2로 축소 (왜냐면 프레임의 크기가 클 경우 연산시간이 증가하므로)
			resize(frameOriginalMat, frame, Size(480, 360), 0, 0, CV_INTER_NN);
			//resize(frameOriginalMat, last_frame, Size(480, 360), 0, 0, CV_INTER_NN);
		}
		catch (cv::Exception& e) {
			// 에러 출력
			std::cerr << "프레임 축소에 실패했기에, 해당 프레임을 무시합니다." << e.err << std::endl;
			isFrameValid = false;
		}

		// 프레임 크기 축소에 성공한 경우 얼굴인식
		if (isFrameValid) {
			try {
				frameGone = true;
				// 프레임을 그레이 스케일 및 이퀄라이즈 처리
				Mat grayframe;
				cvtColor(frame, grayframe, CV_BGR2GRAY);
				equalizeHist(grayframe, grayframe);


				// 얼굴인식 템플릿을 이용하여 얼굴인식
				face_classifier.detectMultiScale(
					grayframe, faces,
					FACE_SEARCH_SCALE,
					MERGE_DETECTED_GROUP_CNTS,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
				);


				//일단 기본이 face!!
				for (int i = 0; i < faces.size(); i++) {

					
					// 얼굴인식 사각형 틀 출력
					lb.x = faces[i].x + faces[i].width;
					lb.y = faces[i].y + faces[i].height;
					
					tr.x = faces[i].x;
					tr.y=faces[i].y;
	
				}

				/////////////////////////////////////////////
				//처음 얼굴 따졌을때 한번만 진행함
				if (isFirst == false &&
					faces.size() != 0) {

					isFirst = true;//처음 영상을 저장

					face_classifier.detectMultiScale(
						first_frame, first_face,
						FACE_SEARCH_SCALE,
						MERGE_DETECTED_GROUP_CNTS,
						CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
						Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
					);

					//resize(frameOriginalMat, first_frame, Size(480, 360), 0, 0, CV_INTER_NN);

					for (int i = 0; i < faces.size(); i++) {
						
						first_lb.x = faces[i].x + faces[i].width;
						first_lb.y = faces[i].y + faces[i].height;

						first_tr.x = faces[i].x;
						first_tr.y = faces[i].y;

						rectangle(frame, first_lb, first_tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.
						//AfxMessageBox(_T("처음 얼굴의 위치"));
					}
				}

				//////////////////////last  영상
				if (isFirst == true &&
					faces.size() != 0) {

					//이전 영상을 저장하기 위함

					face_classifier.detectMultiScale(
						last_frame, last_face,
						FACE_SEARCH_SCALE,
						MERGE_DETECTED_GROUP_CNTS,
						CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
						Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
					);


					//resize(frameOriginalMat, last_frame, Size(480, 360), 0, 0, CV_INTER_NN);

					for (int i = 0; i < faces.size(); i++) {

						last_lb.x = faces[i].x + faces[i].width;
						last_lb.y = faces[i].y + faces[i].height;

						last_tr.x = faces[i].x;
						last_tr.y = faces[i].y;


						rectangle(frame, last_lb, last_tr, Scalar(255, 0, 0), FACE_FRAME_THICKNESS, 4, 0);//프레임에 마지막 얼굴을 표시합니다.
						//AfxMessageBox(_T("마지막"));
					}

				}


				if (faces.size() == 0) {

					cnt++;

					if (cnt >= 40) {

						//grayframe의 마지막 얼굴의 위치를 보여줘야되는 것임
						rectangle(grayframe, last_lb, last_tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 마지막 얼굴을 표시합니다.

						

						rectangle(frame, first_lb, first_tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.
						rectangle(frame, last_lb, last_tr, Scalar(255, 0, 0), FACE_FRAME_THICKNESS, 4, 0);//프레임에 마지막 얼굴을 표시합니다.
						

						 //// 윈도우에 결과 프레임 출력
						imshow("original", frame);
						//imshow("snooze", last_frame);
						AfxMessageBox(_T("다슬이가 졸고있습니다. 깨워주세요!!!"));

						destroyAllWindows();
						return;

					}
						
				}
				else {//frame이 잡히긴 했을 경우
					
					
					cnt--;

					if (cnt <= 0) {
						cnt = 0;
					}

					rectangle(frame, first_lb, first_tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.

					//// 윈도우에 결과 프레임 출력																			
					imshow("original", frame);
					//imshow("snooze", last_frame);

				}

				/////////////////////////////////////
				//n이 0일때엔 비교 ㄴㄴ
					
				//frame 모션 벡터 전체 출력
				if (nc % 2 == 0) {//n이 짝수일때에는 2->1 을보고	
					for (int i = 0; i < aviHeight; i++) {
						for (int j = 0; j < aviWidth; j++)
						{//BRG순서
							aviBuffer2[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];
							aviBuffer2[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
							aviBuffer2[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
							aviBuffer1[i][j].rgbBlue = 0;
							aviBuffer1[i][j].rgbGreen =0;
							aviBuffer1[i][j].rgbRed = 0;
						}
					}
				}
				else {//n이 홀수일때에는 1->2 을봐야지
					for (int i = 0; i < aviHeight; i++) {
						for (int j = 0; j < aviWidth; j++)
						{//BRG순서
							aviBuffer1[i][j].rgbBlue = frame.at<Vec3b>(i, j)[0];
							aviBuffer1[i][j].rgbGreen = frame.at<Vec3b>(i, j)[1];
							aviBuffer1[i][j].rgbRed = frame.at<Vec3b>(i, j)[2];
							aviBuffer2[i][j].rgbBlue = 0;
							aviBuffer2[i][j].rgbGreen = 0;
							aviBuffer2[i][j].rgbRed = 0;
							
						}
					}
				}

				fprintf(file, value3SS);

				if (nc >= 1) {//n이 1일때부터 3SS시작
					for (int i = 0; i <block_h_n; i++)//블록의 높이만큼
					{
						for (int j = 0; j < block_w_n; j++)//블록의 너비만큼
						{
							ArrValue av;
							av = On3SS(16 * (j + 1) - 8, 16 * (i + 1) - 8, av, 3, nc);
							value3SS.Format(_T("(%3d, %3d) "), av.x_dif, av.y_dif);
							fprintf(file, value3SS);
						}

					}
					
					value3SS.Format(_T("-%d & %d 프레임의 모션변화값- \n\n"), nc - 1, nc);//n-1과 n번째 프레임의 모션변화값을 출력합니다=> 텍파로 out
					fprintf(file, value3SS);
				}

				

			}//try close
			catch (cv::Exception& e) {
				cerr << "얼굴인식 처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
				
			}

		}//프레임 잡기에 성공한 경우

		int keyCode = waitKey(30);

		// esc 키가 눌리면 프레임 캡쳐 종료
		if (keyCode == 27) {
			break;
		}

	}//exit loop while

	
	return;//종료합니다
}



//필요없음
void CMFCApplication1_0605projectView::OnDsEdgedector()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	//// TODO: 여기에 명령 처리기 코드를 추가합니다.
	//Mat resultHandImg, tmpImg, originalHandImg;
	//VideoCapture video(0);

	//namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	//namedWindow("original_hand_image", CV_WINDOW_AUTOSIZE);
	//namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	//namedWindow("edge_detect_image", CV_WINDOW_AUTOSIZE);
	//namedWindow("median_edge_detect_image", CV_WINDOW_AUTOSIZE);
	////결과창
	//Mat resultHandCenterImg;
	//Mat medianImg, edgeImg;
	////medianImg가 메디안 적용 후 엣지디텤결과담을넘
	////endeImg가 엣지디텤결과담을넘
	////변수 선언
	//while (true) {//반복
	//	video >> tmpImg;//그냥 정보 임시저장용
	//	video >> resultHandCenterImg;//손바닥 인식 결과 저장용
	//	try {
	//		cvtColor(tmpImg, resultHandImg, CV_BGR2YCrCb);
	//		//피부 색 범위 설정
	//		inRange(resultHandImg, Scalar(0, 133, 77), Scalar(255, 173, 127), resultHandImg);

	//		//8bit 단일채널?
	//		originalHandImg = (resultHandImg.size(), CV_8UC3, Scalar(0));
	//		edgeImg = onRGBToHSI(~tmpImg);
	//		tmpImg = onMedian(tmpImg);
	//		medianImg = onRGBToHSI(~tmpImg);
	//		add(edgeImg, Scalar(0), originalHandImg, resultHandImg);
	//		//tmpImg=onRGBToHSI(tmpImg);
	//		//엣지 디텤
	//		cvtColor(medianImg, medianImg, CV_BGR2GRAY);
	//		cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
	//		//그레이 스케일로 변경
	//		imshow("median_edge_detect_image", medianImg);
	//		imshow("edge_detect_image", edgeImg);
	//		imshow("result_hand_image", resultHandImg);
	//		imshow("original_hand_image", originalHandImg);
	//		erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
	//		double radius;
	//		//손바닥 중심 점 반환
	//		Point center = getHandCenter(resultHandImg, radius);
	//		//원으로 손 표시
	//		circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
	//		circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
	//		imshow("handcenter_image", resultHandCenterImg);
	//	}
	//	catch (Exception& e) {
	//		cerr << "처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
	//	}

	//	if (waitKey(27) == 27)
	//		break;
	//}
	////메모리 해제
	//video.release();
	//tmpImg.release();
	//resultHandImg.release();
	//originalHandImg.release();
	//destroyAllWindows();

}

/////////////////////////////////손따기

void DrawLabelingImage(Mat image, int(* connectedLabels)[4]);
Mat onSobel(Mat rgbImg);
Mat onGrayMedian(Mat rgbImg);
Mat EdgeImprove(Mat image);

Mat onGrayMedian(Mat rgbImg) {
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	float table[9] = { 0 };
	for (int i = 1; i < imgHeight - 1; i++) {
		for (int j = 1; j < imgWidth - 1; j++) {
			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					table[(k + 1) * 3 + (l + 1)] = rgbImg.at<uchar>(i + k, j + l);
				}
			}
			for (int k = 0; k < 9; k++) {
				for (int l = k; l < 9; l++) {
					if (table[k] > table[l]) {
						float trash = table[k];
						table[k] = table[l];
						table[l] = trash;
					}
				}
			}
			rgbImg.at<uchar>(i, j) = table[4];
		}
	}
	return rgbImg;
}
Mat EdgeImprove(Mat image) {//픽셀 키우기(유닛 묶을때 잘묶이게 하려고)
	int imgHeight = image.rows;
	int imgWidth = image.cols;
	int flag = 6;
	for (int i = flag; i < imgHeight; i++) {
		for (int j = flag; j < imgWidth; j++) {
			if (image.at<uchar>(i, j) < 128) {
				//왼쪽 위 자신을 제외한 24칸을 자신의 색으로 채움
				for (int k = -flag; k < 1; k++) {
					for (int l = -flag; l < 1; l++) {
						image.at<uchar>(i + k, j + l) = image.at<uchar>(i, j);
					}
				}
			}
		}
	}
	return image;
}

void DrawLabelingImage(Mat img_gray, int(* connectedLabels)[4]) {//아웃풋 담을 2차배열 [2][4]
	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);
	int sumOfBigLables = -1;//초기값 -1
	int **BigLabels = new int*[numOfLables];
	//작은거 거르고 큰거만 남긴 배열 일단 몇개남을지 모르니 작은거 포함한 갯수로 만들어놈
	//int connectedLabels[2][4];
	//큰거중에 수평인?(y값 비슷한)애들 연결
	//[4]가 라벨의 중점 x, y, 라벨에 네모박스 쳤을때 height, width 순서임
	for (int i = 0; i < numOfLables; i++)
		BigLabels[i] = new int[4];
	//[4]가 라벨의 중점 x, y, 라벨에 네모박스 쳤을때 height, width 순서임

	//라벨링된 이미지중 특정 라벨을 컬러로 표현해주기 이 기능은 안쓸듯?ㅎ
	/*for (int y = 0; y<img_labels.rows; ++y) {

	int *label = img_labels.ptr<int>(y);
	Vec3b* pixel = img_color.ptr<Vec3b>(y);


	for (int x = 0; x < img_labels.cols; ++x) {


	if (label[x] == 3) {
	pixel[x][2] = 0;
	pixel[x][1] = 255;
	pixel[x][0] = 0;
	}
	}
	}
	*/

	//라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 6 || height < 6) {
			continue;//일정크기 이하면 무시
		}
		
		if (sumOfBigLables == -1) {
			sumOfBigLables++;
			continue;
		}

		sumOfBigLables++;

		int x = centroids.at<double>(j, 0); //중심좌표
		int y = centroids.at<double>(j, 1);

		BigLabels[sumOfBigLables][0] = x;
		BigLabels[sumOfBigLables][1] = y;
		BigLabels[sumOfBigLables][2] = width;
		BigLabels[sumOfBigLables][3] = height;

		circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		rectangle(img_color, Point(left, top), Point(left + width, top + height),
			Scalar(0, 0, 255), 1);

		putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
			1, Scalar(255, 0, 0), 2);
	}
	for (int i = 0; i <= sumOfBigLables; i++) {
		for (int j = i + 1; j <= sumOfBigLables; j++) {
			if ((BigLabels[i][1] >= BigLabels[j][1] - (BigLabels[j][3] / 2)) &&
				(BigLabels[i][1] <= BigLabels[j][1] + (BigLabels[j][3] / 2))) {
				connectedLabels[0][0] = BigLabels[i][0];
				connectedLabels[0][1] = BigLabels[i][1];
				connectedLabels[0][2] = BigLabels[i][2];
				connectedLabels[0][3] = BigLabels[i][3];
				connectedLabels[1][0] = BigLabels[j][0];
				connectedLabels[1][1] = BigLabels[j][1];
				connectedLabels[1][2] = BigLabels[j][2];
				connectedLabels[1][3] = BigLabels[j][3];
				//
				circle(img_color, Point(connectedLabels[0][0], connectedLabels[0][1]), 10, Scalar(0, 255, 0), 1);
				circle(img_color, Point(connectedLabels[1][0], connectedLabels[1][1]), 10, Scalar(0, 255, 0), 1);
				line(img_color,
					Point(connectedLabels[0][0], connectedLabels[0][1]),
					Point(connectedLabels[1][0], connectedLabels[1][1]),
					Scalar(0, 0, 255), 2);
			}
		}
	}

	namedWindow("Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Labeling Image", img_color);                        // Show our image inside it

	for (int i = 0; i < numOfLables; i++)
		delete[] BigLabels[i];
	delete[] BigLabels;
	//메모리 해제
}
Mat onSobel(Mat rgbImg) {
	float** hueBuffer;
	float** satuBuffer;
	float** intenBuffer;
	float** sobelBuffer;
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	hueBuffer = new float*[imgHeight];//임의수정
	satuBuffer = new float*[imgHeight];//임의수정
	intenBuffer = new float*[imgHeight];//임의수정
	sobelBuffer = new float*[imgHeight];//임의수정
	for (int i = 0; i < imgHeight; i++) {
		hueBuffer[i] = new float[imgWidth];
		satuBuffer[i] = new float[imgWidth];
		intenBuffer[i] = new float[imgWidth];
		sobelBuffer[i] = new float[imgWidth];
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float r = rgbImg.ptr<BYTE>(i, j)[2] / 255.0;
			float g = rgbImg.ptr<BYTE>(i, j)[1] / 255.0;
			float b = rgbImg.ptr<BYTE>(i, j)[0] / 255.0;

			intenBuffer[i][j] = (1.0 / 3.0)*(r + g + b);
			float minRGB;
			if (r > g) {
				minRGB = g;
				if (minRGB > b)
					minRGB = b;
			}
			else {
				minRGB = r;
				if (minRGB > b)
					minRGB = b;
			}
			satuBuffer[i][j] = 1.0 - (3.0 / (float)(r + g + b))*(minRGB);
			hueBuffer[i][j] = acos(((1.0 / 2.0)*((r - g) + (r - b))) / (sqrt((r - g)*(r - g) + (r - b)*(g - b))));
			hueBuffer[i][j] *= 57.29577951;
			if (b > g)
				hueBuffer[i][j] = 360 - hueBuffer[i][j];
			if (hueBuffer[i][j] < 0)
				hueBuffer[i][j] += 360;
			else if (hueBuffer[i][j] > 360)
				hueBuffer[i][j] -= 360;
		}
	}
	for (int i = 1; i < imgHeight - 1; i++) {
		for (int j = 1; j < imgWidth - 1; j++) {
			float Sx = intenBuffer[i - 1][j + 1] +
				2 * intenBuffer[i][j + 1] +
				intenBuffer[i + 1][j + 1] -
				intenBuffer[i - 1][j - 1] -
				2 * intenBuffer[i][j - 1] -
				intenBuffer[i + 1][j - 1];
			float Sy = intenBuffer[i - 1][j - 1] +
				2 * intenBuffer[i - 1][j] +
				intenBuffer[i - 1][j + 1] -
				intenBuffer[i + 1][j - 1] -
				2 * intenBuffer[i + 1][j] -
				intenBuffer[i + 1][j + 1];
			float Sxy = sqrt((float)Sx*Sx + (float)Sy*Sy)*255.0;
			if (Sxy < 80) {//96
				sobelBuffer[i][j] = 0;
			}
			//else if (Sxy >= 96 && Sxy <= 160) {
			//   sobelBuffer[i][j] = Sxy;
			//}
			else {
				sobelBuffer[i][j] = 255.0;
			}
		}
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float H = hueBuffer[i][j];
			float S = satuBuffer[i][j];
			float I = sobelBuffer[i][j];
			float r = 0;
			float g = 0;
			float b = 0;
			if (0 <= H && H < 120) {
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 - S);
				r = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = 1 - (r + b);
			}
			else if (120 <= H&&H < 240) {
				H = H - 120;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				g = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				r = (1.0 / 3.0)*(float)(1 - S);
				b = 1 - (r + g);
			}
			else if (240 <= H&&H <= 360) {
				H = H - 240;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 + (float)((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = (1.0 / 3.0)*(float)(1 - S);
				r = 1 - (g + b);
			}
			rgbImg.ptr<BYTE>(i, j)[2] = 3.0*r*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[1] = 3.0*g*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[0] = 3.0*b*I*255.0;
		}
	}
	for (int i = 0; i < imgHeight; i++) {
		delete[] hueBuffer[i];
		delete[] satuBuffer[i];
		delete[] intenBuffer[i];
		delete[] sobelBuffer[i];
	}
	delete[] hueBuffer;
	delete[] satuBuffer;
	delete[] intenBuffer;
	delete[] sobelBuffer;
	return rgbImg;
}

void CMFCApplication1_0605projectView::OnDsLabel()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	VideoCapture video(0);
	Mat tmpImg;//원본임, 후에 연산시 여기저기 많이 쓰일거임
	Mat textureImg;//텍스쳐
	Mat textureMask;//텍스쳐 마스크
	Mat edgeImg;//엣지
	Mat result;//결과
	while (true) {//반복
		video >> tmpImg;//원본을 받아온당
		resize(tmpImg, tmpImg, Size(480, 320));
		try {
			namedWindow("original", WINDOW_AUTOSIZE);
			imshow("original", tmpImg);
			//캡쳐해낸 원본 출력
			cvtColor(tmpImg, textureImg, CV_BGR2YCrCb);
			//YCrCb 스페이스로 변경
			inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);
			//정해진 피부색 범위로 얼굴, 손 인식 결과는 grayscale 이미지임
			threshold(textureImg, textureMask, 127, 255, THRESH_BINARY);
			//결과로 나온 grayscale 이미지를 threshold해서 뽑아냄, 후에 마스크로 사용할거임
			//////////////////////////////////////////////////////////////////////////////////////////////
			//add(tmpImg, Scalar(0), textureImg, textureMask);
			//위에서 뽑은 결과를 마스크로 이용해 피부 이외에 부분은 검은색으로 칠해버림
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//그담에 피부를 흰색으로 칠해 결국엔 피부=흰색, 그외=검은색 으로 변경///////////////////////// 근데 이 과정이 필요가 없없음;
			//////////////////////////////////////////////////////////////////////////////////////////////
			result = ~onGrayMedian(textureMask);
			//그냥 마스크 뽑아낸거 자체가 피부랑 나머지 나눠주는 바이너리 이미지임 
			//이래서 이걸 넘겨서 median 함수를 이용해 튀는값 없애준당

			edgeImg = onSobel(tmpImg);
			//처음으로 돌아가 원본사진을 엣지디텤해준다
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//이 엣지 뽑아낸걸 그레이 스케일로 변경한다
			threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//쓰레시홀드 해서 엣지=흰색, 그외=검은색 으로 변경
			edgeImg = EdgeImprove(~edgeImg);
			//엣지에 있는 픽셀들 크기 키워서 서로 잘 이어지도록 해준다

			add(~edgeImg, Scalar(0), result, ~edgeImg);
			//이제 엣지랑 얼굴+손 뽑은거 합쳐서 얼굴, 손 구분되도록 해서
			/////////////////////DrawLabelingImage(~result);이거 주석풀어야해ㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐㅐ
			//라벨들 뽑아낸당, 박스치고 그런건 요 함수 안에서 알아서 해줌 ㅎ
			/*namedWindow("sibal13", WINDOW_AUTOSIZE);
			imshow("sibal13", tmpImg);*/
		}
		catch (Exception& e) {
			cerr << "처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
		} 

		if (waitKey(27) == 27)
			break;
	}
	//메모리 해제
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();
}


void CMFCApplication1_0605projectView::OnDsRotateface()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	VideoCapture video(0);
	Mat tmpImg;//원본임, 후에 연산시 여기저기 많이 쓰일거임
	Mat textureImg;//텍스쳐
	Mat textureMask;//텍스쳐 마스크
	Mat edgeImg;//엣지
	Mat result;//결과
	while (true) {//반복
		video >> tmpImg;//원본을 받아온당
		resize(tmpImg, tmpImg, Size(480, 320));
		try {
			namedWindow("original", WINDOW_AUTOSIZE);
			imshow("original", tmpImg);
			//캡쳐해낸 원본 출력
			cvtColor(tmpImg, textureImg, CV_BGR2YCrCb);
			//YCrCb 스페이스로 변경
			inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);
			//정해진 피부색 범위로 얼굴, 손 인식 결과는 grayscale 이미지임
			threshold(textureImg, textureMask, 127, 255, THRESH_BINARY);
			int connectedLabels[2][4] = { 0 };

			DrawLabelingImage(~EdgeImprove(textureMask),connectedLabels);
			//결과로 나온 grayscale 이미지를 threshold해서 뽑아냄, 후에 마스크로 사용할거임
			//////////////////////////////////////////////////////////////////////////////////////////////
			//add(tmpImg, Scalar(0), textureImg, textureMask);
			//위에서 뽑은 결과를 마스크로 이용해 피부 이외에 부분은 검은색으로 칠해버림
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//그담에 피부를 흰색으로 칠해 결국엔 피부=흰색, 그외=검은색 으로 변경///////////////////////// 근데 이 과정이 필요가 없없음;
			//////////////////////////////////////////////////////////////////////////////////////////////
			/*result = ~onGrayMedian(textureMask);
			//그냥 마스크 뽑아낸거 자체가 피부랑 나머지 나눠주는 바이너리 이미지임 
			//이래서 이걸 넘겨서 median 함수를 이용해 튀는값 없애준당

			edgeImg = onSobel(tmpImg);
			//처음으로 돌아가 원본사진을 엣지디텤해준다
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//이 엣지 뽑아낸걸 그레이 스케일로 변경한다
			threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//쓰레시홀드 해서 엣지=흰색, 그외=검은색 으로 변경
			edgeImg = EdgeImprove(~edgeImg);
			//엣지에 있는 픽셀들 크기 키워서 서로 잘 이어지도록 해준다

			add(~edgeImg, Scalar(0), result, ~edgeImg);
			//이제 엣지랑 얼굴+손 뽑은거 합쳐서 얼굴, 손 구분되도록 해서
			DrawLabelingImage(~result);
			//라벨들 뽑아낸당, 박스치고 그런건 요 함수 안에서 알아서 해줌 ㅎ
			/*namedWindow("sibal13", WINDOW_AUTOSIZE);
			imshow("sibal13", tmpImg);*/
		}
		catch (Exception& e) {
			cerr << "처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}
	//메모리 해제
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();


}



bool handFlag = false;
Mat DrawLabelingImageHand(Mat img_gray, int* handLabels, bool& substracted, int* resultHandLabels)
{
	//아웃풋 담을 2차배열 [2][4]

	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);
	int sumOfBigLables = -1;//초기값 -1
	int **BigLabels = new int*[numOfLables];
	//작은거 거르고 큰거만 남긴 배열 일단 몇개남을지 모르니 작은거 포함한 갯수로 만들어놈
	//int connectedLabels[2][4];
	//큰거중에 수평인?(y값 비슷한)애들 연결
	//[4]가 라벨의 중점 x, y, 라벨에 네모박스 쳤을때 height, width 순서임
	for (int i = 0; i < numOfLables; i++)
		BigLabels[i] = new int[4];
	//[4]가 라벨의 중점 x, y, 라벨에 네모박스 쳤을때 height, width 순서임

	//라벨링된 이미지중 특정 라벨을 컬러로 표현해주기 이 기능은 안쓸듯?ㅎ
	/*for (int y = 0; y<img_labels.rows; ++y) {

	int *label = img_labels.ptr<int>(y);
	Vec3b* pixel = img_color.ptr<Vec3b>(y);


	for (int x = 0; x < img_labels.cols; ++x) {


	if (label[x] == 3) {
	pixel[x][2] = 0;
	pixel[x][1] = 255;
	pixel[x][0] = 0;
	}
	}
	}
	*/

	//라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 20 || height < 20) {
			continue;//일정크기 이하면 무시
		}
		/*CString str;
		str.Format("%d", left, top, width);
		AfxMessageBox(str);
*/
		if (sumOfBigLables == -1) {//처음 들어온놈(머리) 무시
			sumOfBigLables++;
			continue;
		}


		int x = centroids.at<double>(j, 0); //중심좌표
		int y = centroids.at<double>(j, 1);

		BigLabels[sumOfBigLables][0] = left;
		BigLabels[sumOfBigLables][1] = top;
		BigLabels[sumOfBigLables][2] = width;
		BigLabels[sumOfBigLables][3] = height;
		sumOfBigLables++;//카운트값 증가

		//circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		//rectangle(img_color, Point(left, top), Point(left + width, top + height),
		//	Scalar(0, 0, 255), 1);

		//putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
		//	1, Scalar(255, 0, 0), 2);
	}
	if (sumOfBigLables > 1)
		sumOfBigLables = 1;
	//2개 이상은 필요없으니 최대 2개만 뽑을고양시 일산서구 대화동 2026 우리집^^

	for (int i = 0; i < sumOfBigLables; i++) {
		handLabels[0] = BigLabels[i][0];
		handLabels[1] = BigLabels[i][1];
		handLabels[2] = BigLabels[i][2];
		handLabels[3] = BigLabels[i][3];
		//rectangle(img_color, Point(handLabels[0], handLabels[1]), Point(handLabels[0]+ handLabels[2], handLabels[1]+ handLabels[3]),
		//	Scalar(0, 0, 255), 1);
	}
	namedWindow("Hand Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Hand Labeling Image", img_color);
	frameCnt++;// Show our image inside it
	if (frameCnt > 4) {
		if (handFlag == false && sumOfBigLables != 0)
		{

			Mat handLabeled;
			Rect rect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
			hand3ss = img_color(rect);//Rect(handLabels[0], handLabels[1], handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
			resultHandLabels[0] = handLabels[0];
			resultHandLabels[1] = handLabels[1];
			resultHandLabels[2] = handLabels[2];
			resultHandLabels[3] = handLabels[3];

			handFlag = true;// Show our image inside it
	//		namedWindow("3", WINDOW_AUTOSIZE);             // Create a window for display
	//		imshow("3", hand3ss);                        // Show our image inside it

		}

	}
	//Rect newRect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
	//Mat newHand = img_color(newRect);
	//새로운 핸드
	for (int i = 0; i < numOfLables; i++)
		delete[] BigLabels[i];
	delete[] BigLabels;
	//메모리 해제
	return img_color;//새로운 화면 반환
}

void CMFCApplication1_0605projectView::OnDsHanddetector32784()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	VideoCapture video(0);
	Mat tmpImg;//원본임, 후에 연산시 여기저기 많이 쓰일거임
	Mat textureImg;//텍스쳐
	Mat textureMask;//텍스쳐 마스크
	Mat edgeImg;//엣지
	Mat result;//결과
	Mat tmp;
	Mat newHand;
	int resultHandLabels[4] = { 0 };


	/////////////////
	//다슬 3ss
	//처음 손
	Mat firstHand_mat;
	Rect first_hand_rect;
	
	bool isFirstHand = false;//나 처음이니?
	
	 //마지막 손
	Mat lastHand_mat;
	Rect last_hand_rect;


	//flag
	bool isFirstHandd = false;


	while (true) {//반복
		video >> tmp;
		video >> tmpImg;//원본을 받아온당
		resize(tmpImg, tmpImg, Size(480, 320));
		
		try {

			/////////////namedWindow("original", WINDOW_AUTOSIZE);
			/////////////imshow("original", tmpImg);
			//캡쳐해낸 원본 출력
			cvtColor(tmpImg, textureImg, CV_BGR2YCrCb);
			//YCrCb 스페이스로 변경
			inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);
			//정해진 피부색 범위로 얼굴, 손 인식 결과는 grayscale 이미지임
			threshold(textureImg, textureMask, 127, 255, THRESH_BINARY);
			result = ~onGrayMedian(textureMask);
			//그냥 마스크 뽑아낸거 자체가 피부랑 나머지 나눠주는 바이너리 이미지임 
			//이래서 이걸 넘겨서 median 함수를 이용해 튀는값 없애준당

			edgeImg = onSobel(tmpImg);
			//처음으로 돌아가 원본사진을 엣지디텤해준다
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//이 엣지 뽑아낸걸 그레이 스케일로 변경한다
			threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//쓰레시홀드 해서 엣지=흰색, 그외=검은색 으로 변경
			edgeImg = EdgeImprove(~edgeImg);
			//엣지에 있는 픽셀들 크기 키워서 서로 잘 이어지도록 해준다

			add(~edgeImg, Scalar(0), result, ~edgeImg);

			int handLabels[4] = { 0 };
			bool subtracted;
			if (newHand.data != nullptr) {
				//새로 들어올 hand를 위해 초기화 해준당
				newHand.data = nullptr;
			}
			newHand = DrawLabelingImageHand(~result, handLabels, subtracted, resultHandLabels);
			//새로운 핸드

			namedWindow("성공?", WINDOW_AUTOSIZE);             // Create a window for display
			imshow("성공?", hand3ss);                       // Show our image inside it
			namedWindow("성공!", WINDOW_AUTOSIZE);             // Create a window for display
			imshow("성공!", newHand);                       // Show our image inside it
														  //	}//얘는 흑백처럼 보이지만 BGR임(hand3ss)

			if (hand3ss.data != nullptr){
				//처음 들어오는 손이 있는 경우

				if (isFirstHand == false) {
					isFirstHand = true;

					//AfxMessageBox(_T("처음손"));
					Rect rect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
					first_hand_rect = rect;
					rectangle(tmp, first_hand_rect, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.
																						//처음 들어오는 손이 있는 경우 손을 먼저 original화면에서 라벨링해서 보여준다.
					namedWindow("original", WINDOW_AUTOSIZE);
					imshow("original", tmp);
				}
				
			}

			if (handLabels[0]!=NULL
				&& handLabels[1]!=NULL
				&& handLabels[2]!=NULL
				&& handLabels[3]!=NULL) {
				//새로들어온 손이 있기는 할 경우
				Rect rect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
				last_hand_rect = rect;
				//AfxMessageBox(_T("아직손이있엉"));
				rectangle(tmp, first_hand_rect, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);
				rectangle(tmp, last_hand_rect, Scalar(255, 0, 0), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.
				//처음 들어오는 손이 있는 경우 손을 먼저 original화면에서 라벨링해서 보여준다.
				namedWindow("original", WINDOW_AUTOSIZE);
				imshow("original", tmp);

			}
			else {
				//새로 들어온 손이 없는 경우
				AfxMessageBox(_T("이제 손이없엉 ㅠ"));
				rectangle(tmp, last_hand_rect, Scalar(255, 0, 0), FACE_FRAME_THICKNESS, 4, 0);//처음 얼굴은 계속 있어야 되는것임//프레임에 처음 얼굴을 표시합니다.
				rectangle(tmp, first_hand_rect, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);																	  //처음 들어오는 손이 있는 경우 손을 먼저 original화면에서 라벨링해서 보여준다.
				namedWindow("original", WINDOW_AUTOSIZE);
				imshow("original", tmp);
			}
			
		}
		catch (Exception& e) {
			cerr << "처리에 실패했기에, 해당 프레임을 무시합니다." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}//while loop finish

	 //메모리 해제
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();
}
