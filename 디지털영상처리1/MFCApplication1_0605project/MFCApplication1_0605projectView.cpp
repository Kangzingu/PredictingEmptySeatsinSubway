#pragma once
// MFCApplication1_0605projectView.cpp : CMFCApplication1_0605projectView 클래스의 구현
//
#include "stdafx.h"
#include "opencv2/opencv.hpp"

#include "opencv2/objdetect.hpp"//
#include "opencv2/videoio.hpp"//
#include "opencv2/highgui.hpp"//
#include "opencv2/imgproc.hpp"//

using namespace cv;
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFCApplication1_0605project.h"
#endif

#include "MFCApplication1_0605projectDoc.h"
#include "MFCApplication1_0605projectView.h"

#include <vector>
#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;

#define M_PI 3.14159265358979323846
#define WIN_NAME "얼굴인식"
#define FACE_CLASSIFIER_PATH "C:/Users/kanrh/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
#define FACE_SEARCH_SCALE 1.1
#define MERGE_DETECTED_GROUP_CNTS 3
#define FACE_FRAME_WIDTH 50
#define FACE_FRAME_HEIGHT 50
#define FACE_FRAME_THICKNESS 1
#define eyes_cascade_name = "C:/Users/kanrh/Downloads/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
//안구 인식 XML 파일 경로 선언

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
	ON_COMMAND(ID_BACKGROUNDDETECTION, &CMFCApplication1_0605projectView::OnBackgrounddetection)
	ON_COMMAND(ID_HANDDETECTION, &CMFCApplication1_0605projectView::OnHanddetection)
	ON_COMMAND(ID_FACEDETECTION, &CMFCApplication1_0605projectView::OnFacedetection)
	ON_COMMAND(ID_EDGEDETECTION, &CMFCApplication1_0605projectView::OnEdgedetection)
	ON_COMMAND(ID_LABELING, &CMFCApplication1_0605projectView::OnLabeling)
	ON_COMMAND(ID_HANDLABELING, &CMFCApplication1_0605projectView::OnHandlabeling)
	ON_COMMAND(ID_BACKGROUNDSUBWAY, &CMFCApplication1_0605projectView::OnBackgroundsubway)
END_MESSAGE_MAP()
//Mat getHandMask1(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158);
// CMFCApplication1_0605projectView 생성/소멸
Point getHandCenter(const Mat& mask, double& radius);
Mat DrawLabelingImage(Mat image);
Mat onSobel(Mat rgbImg);
Mat onGrayMedian(Mat rgbImg);
Mat onMedian(Mat rgbImg);
Mat EdgeImprove(Mat image);
void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected);
Mat DrawLabelingImageHand(Mat img_gray, int*handLabels, bool& substracted);

Mat hand3ss;
bool handFlag = false;
//Mat input3ss;
void On3ss(Mat frame);
RGBQUAD** aviBuffer1;
RGBQUAD** aviBuffer2;
int aviHeight;
int aviWidth;
int cnt = 0;
int allowCnt = 0;
int frameCnt = 0;

class ArrValue {
public:
	int x_dif = 0;
	int y_dif = 0;
};
ArrValue On3SS(int x, int y, ArrValue av, int number, int n);
double calculate_subtraction(int x, int y, int x_dif, int y_dif, int n);

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
		if (frame.data == nullptr)
			break;
		imshow("video", frame);
		if (waitKey(30) >= 0)
			break;
	}
	//AfxMessageBox("Completed");
}
void CMFCApplication1_0605projectView::OnBackgrounddetection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;
	// Init background substractor
	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	//MOG2 함수를 이용한 배경 추출
	Mat foregroundMask, backgroundImage, foregroundImg;
	//////////////////////////VideoCapture cap(0);
	int sumWhite = 0; int sumBlack = 0;
	CString result;
	for (;;) {
		Mat img;
		Capture >> img;
		//프레임 불러오기
		if (img.data == nullptr)
			break;
		resize(img, img, Size(480, 320));
		if (foregroundMask.empty()) {
			foregroundMask.create(img.size(), img.type());
		}
		bg_model->apply(img, foregroundMask, true ? -1 : 0);
		GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
		//전처리로 블러링 적용
		threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
		foregroundImg = Scalar::all(0);
		img.copyTo(foregroundImg, foregroundMask);
		bg_model->getBackgroundImage(backgroundImage);
		//배경이미지를 가져와
		imshow("foreground mask", foregroundMask);
		//우리가 원하는 흑백 물체감지
		imshow("foreground image", foregroundImg);
		int key6 = waitKey(40);
		if (!backgroundImage.empty()) {
			imshow("background image", backgroundImage);
			int key5 = waitKey(40);
		}
		imshow("original image", img);
		for (int i = 0; i < 320; i++) {
			for (int j = 0; j < 480; j++) {
				if (foregroundMask.ptr<BYTE>(i, j)[0] == 0)
					sumBlack++;
				else
					sumWhite++;
			}
		}
		CString paste;
		float sum = (float)sumWhite / (sumBlack + sumWhite);
		//상대적 비율을 구해
		paste.Format("(%f)\n", sum);
		result += paste;
		//출력 준비
		CStdioFile file;
		file.Open(_T("FILE.TXT"), CFile::modeCreate | CFile::modeWrite);
		//쓰기전용으로 열고, 혹시나 파일명 못찾으면 해당 파일 명으로 새로 만들고
		file.WriteString(result);
		file.Close();
	}
}
void CMFCApplication1_0605projectView::OnHanddetection()
{
	Mat resultHandImg, tmpImg, originalHandImg;
	VideoCapture video(0);
	namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("original_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	//결과창
	Mat resultHandCenterImg;
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);
	//경로지정
	Mat resultFaceImg;
	vector<Rect> faces;
	//변수 선언
	while (true) {//반복
		video >> tmpImg;//그냥 정보 임시저장용
		video >> resultHandCenterImg;//손바닥 인식 결과 저장용
		bool isFrameValid = true;
		try {
			// 웹 캠 프레임의 원본 크기 저장
			video >> tmpImg;

			// 원본 크기의 1/2로 축소 할수있징만 나는 하지 않을거야 (왜냐면 프레임의 크기가 클 경우 연산시간이 증가)
			resize(tmpImg, resultFaceImg, cv::Size(tmpImg.cols / 1, tmpImg.rows / 1), 0, 0, CV_INTER_NN);
		}
		catch (cv::Exception& e) {
			// 에러 출력
			std::cerr << "프레임 축소에 실패했기에, 해당 프레임을 무시합니다." << e.err << std::endl;
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
				imshow("original image", tmpImg);
				imshow("result_face_image", resultFaceImg);
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

			imshow("result_hand_image", resultHandImg);
			imshow("original_hand_image", originalHandImg);
			erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
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
			//손바닥 중심 점 반환
			Point center = getHandCenter(resultHandImg, radius);
			//원으로 손 표시
			circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
			circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
			imshow("handcenter_image", resultHandCenterImg);
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
	resultHandImg.release();
	originalHandImg.release();
	destroyAllWindows();
}
Mat onMedian(Mat rgbImg) {
	float** hueBuffer;
	float** satuBuffer;
	float** intenBuffer;
	float** mediBuffer;
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	hueBuffer = new float*[imgHeight];//임의수정
	satuBuffer = new float*[imgHeight];//임의수정
	intenBuffer = new float*[imgHeight];//임의수정
	mediBuffer = new float*[imgHeight];//임의수정
	for (int i = 0; i < imgHeight; i++) {
		hueBuffer[i] = new float[imgWidth];
		satuBuffer[i] = new float[imgWidth];
		intenBuffer[i] = new float[imgWidth];
		mediBuffer[i] = new float[imgWidth];
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
	//
	float table[25] = { 0 };
	for (int i = 2; i < imgHeight - 2; i++) {
		for (int j = 2; j < imgWidth - 2; j++) {
			table[0] = intenBuffer[i - 2][j - 2];
			table[1] = intenBuffer[i - 2][j - 1];
			table[2] = intenBuffer[i - 2][j];
			table[3] = intenBuffer[i - 2][j + 1];
			table[4] = intenBuffer[i - 2][j + 2];
			table[5] = intenBuffer[i - 1][j - 2];
			table[6] = intenBuffer[i - 1][j - 1];
			table[7] = intenBuffer[i - 1][j];
			table[8] = intenBuffer[i - 1][j + 1];
			table[9] = intenBuffer[i - 1][j + 2];
			table[10] = intenBuffer[i][j - 2];
			table[11] = intenBuffer[i][j - 1];
			table[12] = intenBuffer[i][j];
			table[13] = intenBuffer[i][j + 1];
			table[14] = intenBuffer[i][j + 2];
			table[15] = intenBuffer[i + 1][j - 2];
			table[16] = intenBuffer[i + 1][j - 1];
			table[17] = intenBuffer[i + 1][j];
			table[18] = intenBuffer[i + 1][j + 1];
			table[19] = intenBuffer[i + 1][j + 2];
			table[20] = intenBuffer[i + 2][j - 2];
			table[21] = intenBuffer[i + 2][j - 1];
			table[22] = intenBuffer[i + 2][j];
			table[23] = intenBuffer[i + 2][j + 1];
			table[24] = intenBuffer[i + 2][j + 2];
			for (int k = 0; k < 25; k++) {
				for (int l = k; l < 25; l++) {
					if (table[k] > table[l]) {
						float trash = table[k];
						table[k] = table[l];
						table[l] = trash;
					}
				}
			}
			mediBuffer[i][j] = table[12];
		}
	}
	//
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float H = hueBuffer[i][j];
			float S = satuBuffer[i][j];
			float I = mediBuffer[i][j];
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
		delete[] mediBuffer[i];
	}
	delete[] hueBuffer;
	delete[] satuBuffer;
	delete[] intenBuffer;
	delete[] mediBuffer;
	return rgbImg;
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
void CMFCApplication1_0605projectView::OnFacedetection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
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
			resize(frameOriginalMat, frame, cv::Size(frameOriginalMat.cols / 2, frameOriginalMat.rows / 2), 0, 0, CV_INTER_NN);
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
				cv::imshow(WIN_NAME, frame);
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
void CMFCApplication1_0605projectView::OnEdgedetection()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	Mat resultHandImg, tmpImg, originalHandImg;
	VideoCapture video(0);
	namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("edge_hand_result_image", CV_WINDOW_AUTOSIZE);
	namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	namedWindow("edge_detect_image", CV_WINDOW_AUTOSIZE);
	namedWindow("median_edge_detect_image", CV_WINDOW_AUTOSIZE);
	//결과창
	Mat resultHandCenterImg;
	Mat medianImg, edgeImg;
	//medianImg가 메디안 적용 후 엣지디텤결과담을넘
	//endeImg가 엣지디텤결과담을넘
	//변수 선언
	while (true) {//반복
		video >> tmpImg;//그냥 정보 임시저장용
		video >> resultHandCenterImg;//손바닥 인식 결과 저장용
		try {
			cvtColor(tmpImg, resultHandImg, CV_BGR2YCrCb);
			//피부 색 범위 설정
			inRange(resultHandImg, Scalar(0, 133, 77), Scalar(255, 173, 127), resultHandImg);

			//8bit 단일채널?
			originalHandImg = (resultHandImg.size(), CV_8UC3, Scalar(0));
			edgeImg = onSobel(~tmpImg);

			tmpImg = onMedian(tmpImg);
			medianImg = onSobel(~tmpImg);
			add(edgeImg, Scalar(0), originalHandImg, resultHandImg);
			//tmpImg=onRGBToHSI(tmpImg);
			//엣지 디텤
			cvtColor(medianImg, medianImg, CV_BGR2GRAY);
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//그레이 스케일로 변경
			imshow("median_edge_detect_image", medianImg);//
			imshow("edge_detect_image", edgeImg);
			imshow("result_hand_image", resultHandImg);
			imshow("edge_hand_result_image", originalHandImg);//손 얼굴 색으로 뽑은거랑 엣지디텍한거 합친당
			erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
			double radius;
			//손바닥 중심 점 반환
			Point center = getHandCenter(resultHandImg, radius);
			//원으로 손 표시
			circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
			circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
			imshow("handcenter_image", resultHandCenterImg);
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
	resultHandImg.release();
	originalHandImg.release();
	destroyAllWindows();
}

void CMFCApplication1_0605projectView::OnLabeling(){
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
	int flag = 5;
	for (int i = flag; i < imgHeight; i++) {
		for (int j = flag; j < imgWidth; j++) {
			if (image.at<uchar>(i, j) < 128) {
				//왼쪽 위 자신을 포함한 24칸을 자신의 색으로 채움
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

void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected)
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
		if (width < 6 || height < 6) {
			continue;//일정크기 이하면 무시
		}

		if (sumOfBigLables == -1) {
			sumOfBigLables++;
			continue;
		}


		int x = centroids.at<double>(j, 0); //중심좌표
		int y = centroids.at<double>(j, 1);

		BigLabels[sumOfBigLables][0] = x;
		BigLabels[sumOfBigLables][1] = y;
		BigLabels[sumOfBigLables][2] = width;
		BigLabels[sumOfBigLables][3] = height;
		sumOfBigLables++;

		/*circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		rectangle(img_color, Point(left, top), Point(left + width, top + height),
			Scalar(0, 0, 255), 1);

		putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
			1, Scalar(255, 0, 0), 2);
	*/
	}
	connected = false;
	if (sumOfBigLables > 4)
		sumOfBigLables = 4;
	//4개 이상이면 4개까지만 검사
	for (int i = 0; i < sumOfBigLables; i++) {
		for (int j = i + 1; j < sumOfBigLables; j++) {
			if ((BigLabels[i][1] >= BigLabels[j][1] - (BigLabels[j][3] / 2)) &&
				(BigLabels[i][1] <= BigLabels[j][1] + (BigLabels[j][3] / 2))) {//연결찾았어
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
				connected = true;//연결했으니 true로 변경
				i = sumOfBigLables - 1;//끝낸다
				j = sumOfBigLables - 1;//"
			}
		}
	}
	namedWindow("Eye Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Eye Labeling Image", img_color);                        // Show our image inside it

	for (int i = 0; i < numOfLables; i++)
		delete[] BigLabels[i];
	delete[] BigLabels;
	//메모리 해제

}
Mat DrawLabelingImage(Mat img_gray) {
	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);


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
	int sumOfBigLables = 0;//초기값 -1
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 20 || height < 20) {
			continue;//일정크기 이하면 무시
		}
		//if (sumOfBigLables == -1) {
		//	sumOfBigLables++;
		//	continue;
		//}
		sumOfBigLables++;
		int x = centroids.at<double>(j, 0); //중심좌표
		int y = centroids.at<double>(j, 1);

		circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		rectangle(img_color, Point(left, top), Point(left + width, top + height),
			Scalar(0, 0, 255), 1);

		putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
			1, Scalar(255, 0, 0), 2);
	}

	namedWindow("Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Labeling Image", img_color);                        // Show our image inside it
	return img_color;
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
			//	sobelBuffer[i][j] = Sxy;
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

void CMFCApplication1_0605projectView::OnHandtracking() {



}


void On3ss(Mat frame) {
	//frame을 받아서 3ss를 진행

	if (frame.data == nullptr)
		AfxMessageBox("Error Video");

	//크기만큼 생성
	aviHeight = frame.rows;
	aviWidth = frame.cols;
	aviBuffer1 = new RGBQUAD*[aviHeight];
	aviBuffer2 = new RGBQUAD*[aviHeight];

	for (int i = 0; i < aviHeight; i++)
	{
		aviBuffer1[i] = new RGBQUAD[aviWidth];
		aviBuffer2[i] = new RGBQUAD[aviWidth];
	}

	//3SS을 위한 준비
	FILE* file = fopen("HAND3SS.txt", "wt");
	CString value3SS;

	//frame 크기의 16분의1을 블록사이즈로 지정
	int hand_w = hand3ss.cols;
	int hand_h = hand3ss.rows;
	int block_w_n = aviWidth / hand_w;
	int block_h_n = aviHeight / hand_h;
	int value = 0;

	//namedWindow("img1", CV_WINDOW_AUTOSIZE);
	//imshow("img1", frame);

	/*
	모든 프레임마다 3ss를 바로 진행

	*/
	if (cnt % 2 == 0)
	{
		for (int i = 0; i < aviHeight; i++) {
			for (int j = 0; j < aviWidth; j++)
			{//BRG순서
				aviBuffer2[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];//여기에 첫 프레임이 저장되어야 함 =>hand3ss
				aviBuffer2[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
				aviBuffer2[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
				aviBuffer1[i][j].rgbBlue = 0;
				aviBuffer1[i][j].rgbGreen = 0;
				aviBuffer1[i][j].rgbRed = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < aviHeight; i++) {
			for (int j = 0; j < aviWidth; j++)
			{//BRG순서
				aviBuffer1[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];//여기에 두번째 프레임이 저장되어야 함=>hand3ss
				aviBuffer1[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
				aviBuffer1[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
				aviBuffer2[i][j].rgbBlue = 0;
				aviBuffer2[i][j].rgbGreen = 0;
				aviBuffer2[i][j].rgbRed = 0;

			}
		}
	}


	//////모든 프레임이 여기서부터 시작
	if (cnt >= 1) {//3ss
		for (int i = 0; i < block_h_n; i++)
		{
			for (int j = 0; j < block_w_n; j++)
			{
				ArrValue av;
				av = On3SS(16 * (j + 1) - 8, 16 * (i + 1) - 8, av, 3, cnt);
				value3SS.Format(_T("(%3d, %3d) "), av.x_dif, av.y_dif);
				fprintf(file, value3SS);
			}

			fprintf(file, "\n");
		}
		value3SS.Format(_T("-%d & %d 프레임의 모션변화값- \n\n"), cnt - 1, cnt);
		fprintf(file, value3SS);
	}
	//n이 0일때엔 비교 ㄴㄴ      //n이 1일때부터 3SS시작이고, 
	//n이 짝수일때에는 2->1 을보고      //n이 홀수일때에는 1->2 을봐야지
	cnt++;
	////////////////////////
	fclose(file);


	//AfxMessageBox("Completed");

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
			//         if (!(x< 0 || x>aviWidth || y <0 || y>aviHeight) && !(x+x2<0 || x+x2>aviWidth|| y+y2<0||y+y2>aviHeight))
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
	/*CString str;
	str.Format("%lf", min);
	AfxMessageBox(str);
	*/if (min == 0) {
	//값이 같을때=>이동하지 않음
		min_idx = 4;

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

double calculate_subtraction(int x, int y, int x_dif, int y_dif, int n) {
	int sum = 0;

	//모든 프레임이 들어왔을 경우, avibuffer2와 avibuffer1을 연산

	for (int i = -8; i < 8; i++)
	{
		for (int j = -8; j < 8; j++)
		{
			if (!((x + x_dif + i) <= 0 || (x + x_dif + i) >= aviWidth || (y + y_dif + j) <= 0 || (y + y_dif + j) >= aviHeight))
			{

				if (n % 2 == 1)//
				{
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbRed);

				}
				else//짝수면
				{
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbBlue - (int)aviBuffer2[y + j][x + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbGreen - (int)aviBuffer2[y + j][x + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbRed - (int)aviBuffer2[y + j][x + i].rgbRed);
				}
			}
			else {

				if (n % 2 == 1)//
				{
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbRed);

				}
				else//짝수면
				{

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
	}

	return sum / 16.0 / 16;
}


void CMFCApplication1_0605projectView::OnHandlabeling()
{
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

	while (true) {//반복
		video >> tmp;
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
			///////////////input3ss = DrawLabelingImage(~result);

			//void DrawLabelingImageHand(Mat img_gray, int(*handLabels)[4], int& numOfHandLabels)
			//void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected)
			//AfxMessageBox(str);

			int eyeLabels[2][4] = { 0 };
			bool successOfEyeLabels = false;
			DrawLabelingImageEye(result, eyeLabels, successOfEyeLabels);
			if (successOfEyeLabels == false) {
				allowCnt++;
				if (allowCnt >= 3) {
					//////////////////////////////////


				//	AfxMessageBox(_T("다음역이 건대입구라는것을 안 형규"));
				}
			}
			else {//커넥션이 있어
				if (allowCnt > 0) {
					allowCnt--;
				}
			}

			////AfxMessageBox(str);

			//int handLabels[4] = { 0 };
			//bool subtracted;
			//if (newHand.data != nullptr) {
			//	//새로 들어올 hand를 위해 초기화 해준당
			//	newHand.data = nullptr;
			//}
			//newHand = DrawLabelingImageHand(~result, handLabels, subtracted, resultHandLabels);

			////	if (resultHandLabels[3] == 0) {
			///*CString str;
			//str.Format("%d",handFlag);
			//AfxMessageBox(str);*/

			////rectangle(result, Point(resultHandLabels[0], resultHandLabels[1]), Point(resultHandLabels[0] + resultHandLabels[2], resultHandLabels[1] + resultHandLabels[3]),
			////	Scalar(255, 0, 0), 1);
			//namedWindow("성공?", WINDOW_AUTOSIZE);             // Create a window for display
			//imshow("성공?", hand3ss);                       // Show our image inside it
			//namedWindow("성공!", WINDOW_AUTOSIZE);             // Create a window for display
			//imshow("성공!", newHand);                       // Show our image inside it
			//											  //	}//얘는 흑백처럼 보이지만 BGR임(hand3ss)

			//if (hand3ss.data != nullptr) {
			//	//여기서부터 한번 손이 잡혔고, 이후의 frame인  newHand를 가지고 3ss를 한다
			//	//만약 모션벡터값이 허용치를 넘는경우 메시지 박스
			//	On3ss(newHand);
			//}

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


void CMFCApplication1_0605projectView::OnBackgroundsubway()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;
	// Init background substractor
	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");

	Mat img = imread("lotte.PNG", IMREAD_COLOR);
	resize(img, img, Size(480, 320));
	//
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
			cvtColor(tmpImg, textureImg, CV_BGR2HSV);
			//YCrCb 스페이스로 변경

			//inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);

			//inRange(textureImg, Scalar(124, 42, 26), Scalar(173, 102, 97), textureImg);

			inRange(textureImg, Scalar(65, 60, 60), Scalar(80, 255, 255), textureImg);

			//정해진 피부색 범위로 얼굴, 손 인식 결과는 grayscale 이미지임
			threshold(textureImg, textureMask, 200, 255, THRESH_BINARY);
			//결과로 나온 grayscale 이미지를 threshold해서 뽑아냄, 후에 마스크로 사용할거임
			//////////////////////////////////////////////////////////////////////////////////////////////
			add(tmpImg, Scalar(0), textureImg, ~textureMask);
			//
			Mat backGround;
			add(img, Scalar(0), backGround, textureMask);
			add(backGround, textureImg, textureImg);
			namedWindow("backGroundChange", WINDOW_AUTOSIZE);
			imshow("backGroundChange", textureImg);
			//위에서 뽑은 결과를 마스크로 이용해 피부 이외에 부분은 검은색으로 칠해버림
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//그담에 피부를 흰색으로 칠해 결국엔 피부=흰색, 그외=검은색 으로 변경///////////////////////// 근데 이 과정이 필요가 없없음;
			//////////////////////////////////////////////////////////////////////////////////////////////
			//result = ~onGrayMedian(textureMask);
			//그냥 마스크 뽑아낸거 자체가 피부랑 나머지 나눠주는 바이너리 이미지임 
			//이래서 이걸 넘겨서 median 함수를 이용해 튀는값 없애준당

			//edgeImg = onSobel(tmpImg);
			//처음으로 돌아가 원본사진을 엣지디텤해준다
			//cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//이 엣지 뽑아낸걸 그레이 스케일로 변경한다
			//threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//쓰레시홀드 해서 엣지=흰색, 그외=검은색 으로 변경
			//edgeImg = EdgeImprove(~edgeImg);
			//엣지에 있는 픽셀들 크기 키워서 서로 잘 이어지도록 해준다

			//add(~edgeImg, Scalar(0), result, ~edgeImg);
			//이제 엣지랑 얼굴+손 뽑은거 합쳐서 얼굴, 손 구분되도록 해서
			///////////////input3ss = DrawLabelingImage(~result);
			//input3ss = DrawLabelingImage(~result);
			//라벨들 뽑아낸당, 박스치고 그런건 요 함수 안에서 알아서 해줌 ㅎ
			/*namedWindow("sibal13", WINDOW_AUTOSIZE);
			imshow("sibal13", tmpImg);*/

			//cvtColor(input3ss, input3ss, CV_GRAY2BGR);

			//On3ss(tmp);
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


