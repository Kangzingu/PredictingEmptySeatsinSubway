
// MFCApplication1_0605project.h : MFCApplication1_0605project ���� ���α׷��� ���� �� ��� ����
//
#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"       // �� ��ȣ�Դϴ�.


// CMFCApplication1_0605projectApp:
// �� Ŭ������ ������ ���ؼ��� MFCApplication1_0605project.cpp�� �����Ͻʽÿ�.
//

class CMFCApplication1_0605projectApp : public CWinAppEx
{
public:
	CMFCApplication1_0605projectApp();


// �������Դϴ�.
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// �����Դϴ�.
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CMFCApplication1_0605projectApp theApp;
