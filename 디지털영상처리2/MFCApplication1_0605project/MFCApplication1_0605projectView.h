
// MFCApplication1_0605projectView.h : CMFCApplication1_0605projectView Ŭ������ �������̽�
//

#pragma once


class CMFCApplication1_0605projectView : public CView
{
protected: // serialization������ ��������ϴ�.
	CMFCApplication1_0605projectView();
	DECLARE_DYNCREATE(CMFCApplication1_0605projectView)

// Ư���Դϴ�.
public:
	CMFCApplication1_0605projectDoc* GetDocument() const;

// �۾��Դϴ�.
public:

// �������Դϴ�.
public:
	virtual void OnDraw(CDC* pDC);  // �� �並 �׸��� ���� �����ǵǾ����ϴ�.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// �����Դϴ�.
public:
	virtual ~CMFCApplication1_0605projectView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ������ �޽��� �� �Լ�
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLoadavi();
	afx_msg void OnFaceDetect();
	afx_msg void OnDsEyeDetect();
	afx_msg void OnDsMotionchange();
	afx_msg void OnDsFacedown();
	afx_msg void OnDsEdgedector();
	
	afx_msg void OnDsLabel();
	afx_msg void OnDsRotateface();
	afx_msg void OnDsHanddetector32784();
};

#ifndef _DEBUG  // MFCApplication1_0605projectView.cpp�� ����� ����
inline CMFCApplication1_0605projectDoc* CMFCApplication1_0605projectView::GetDocument() const
   { return reinterpret_cast<CMFCApplication1_0605projectDoc*>(m_pDocument); }
#endif

