/****** Object:  StoredProcedure [dbo].[getpydatav2]    Script Date: 6/19/2020 10:24:33 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Sam Hartzog
-- Create date: 5/20/20
-- Description:	Get data for python
-- =============================================
ALTER PROCEDURE [dbo].[getLabelsForConditionSetv2]
	  @bindata nchar(11) = '00000000000'
	 ,@SplitSex bit = 0
	 ,@minAge int = 0
	 ,@maxAge int = 100
AS
BEGIN
	IF @SplitSex = 0
		BEGIN
		SELECT CASE WHEN nAge < 10 THEN 5
					WHEN nAge < 20 THEN 15
					WHEN nAge < 30 THEN 25
					WHEN nAge < 40 THEN 35
					WHEN nAge < 50 THEN 45
					WHEN nAge < 60 THEN 55
					WHEN nAge < 70 THEN 65
					WHEN nAge < 80 THEN 75
					WHEN nAge < 90 THEN 85
					ELSE 95 END as Age
			,SUM(CASE nTypeOfCare WHEN 2 THEN 1. ELSE 0. END)/COUNT(*) AS Hospitalized
			,SUM(CASE WHEN (nICU = 1 OR nIntubated = 1) THEN 1. ELSE 0. END)/COUNT(*) as Intubated
			,SUM(CASE dtDeath WHEN '1970-01-01 00:00:00.000' THEN 0. ELSE 1. END)/COUNT(*) as Deceased
			,SUM(CASE nPnemonia WHEN 1 THEN 1. ELSE 0. END)/COUNT(*) as Pneumonia
			,COUNT(*) AS CaseCount
		FROM [dbo].[MX_CaseData_200531]
			WHERE nTestResult = 1 
  			and nDiabetes in (1, 2)
			and nAthsma in (1, 2)
			and nImmunosuppressed in (1, 2)
			and nHypertension in (1, 2)
			and nOtherDisease in (1, 2)
			and nCardiovascularDisease in (1, 2)
			and nKidneyDisease in (1, 2)
			and nTobbacco in (1, 2)
			and nCOPD in (1, 2) 
			and nTypeOfCare <> 99 
			and nPnemonia in (1, 2) 
			and nICU <> 99 
			and nIntubated <> 99
			and dtSymptoms < '01-MAY-2020'
			and nAge >= @minAge
			and nAge <= @maxAge
			and (CASE nPregnancy WHEN 1 THEN '1' ELSE '0' END
			+ CASE nDiabetes WHEN 1 THEN '1' ELSE '0' END
			+ CASE nAthsma WHEN 1 THEN '1' ELSE '0' END
			+ CASE nImmunosuppressed WHEN 1 THEN '1' ELSE '0' END
			+ CASE nHypertension WHEN 1 THEN '1' ELSE '0' END
			+ CASE nOtherDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nCardiovascularDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nObesity WHEN 1 THEN '1' ELSE '0' END
			+ CASE nKidneyDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nTobbacco WHEN 1 THEN '1' ELSE '0' END
			+ CASE nCOPD WHEN 1 THEN '1' ELSE '0' END) = @bindata
		GROUP BY CASE WHEN nAge < 10 THEN 5
					WHEN nAge < 20 THEN 15
					WHEN nAge < 30 THEN 25
					WHEN nAge < 40 THEN 35
					WHEN nAge < 50 THEN 45
					WHEN nAge < 60 THEN 55
					WHEN nAge < 70 THEN 65
					WHEN nAge < 80 THEN 75
					WHEN nAge < 90 THEN 85
					ELSE 95 END
		ORDER BY Age
		END
	ELSE
		BEGIN
		SELECT CASE WHEN nAge < 10 THEN 5
					WHEN nAge < 20 THEN 15
					WHEN nAge < 30 THEN 25
					WHEN nAge < 40 THEN 35
					WHEN nAge < 50 THEN 45
					WHEN nAge < 60 THEN 55
					WHEN nAge < 70 THEN 65
					WHEN nAge < 80 THEN 75
					WHEN nAge < 90 THEN 85
					ELSE 95 END as Age
			,CASE nSex WHEN 1 THEN 'Female' ELSE 'Male' END as Sex
			,SUM(CASE nTypeOfCare WHEN 2 THEN 1. ELSE 0. END)/COUNT(*) AS Hospitalized
			,SUM(CASE WHEN (nICU = 1 OR nIntubated = 1) THEN 1. ELSE 0. END)/COUNT(*) as Intubated
			,SUM(CASE dtDeath WHEN '1970-01-01 00:00:00.000' THEN 0. ELSE 1. END)/COUNT(*) as Deceased
			,SUM(CASE nPnemonia WHEN 1 THEN 1. ELSE 0. END)/COUNT(*) as Pneumonia
			,COUNT(*) AS CaseCount
		FROM [dbo].[MX_CaseData_200531]
			WHERE nTestResult = 1 
  			and nDiabetes in (1, 2)
			and nAthsma in (1, 2)
			and nImmunosuppressed in (1, 2)
			and nHypertension in (1, 2)
			and nOtherDisease in (1, 2)
			and nCardiovascularDisease in (1, 2)
			and nKidneyDisease in (1, 2)
			and nTobbacco in (1, 2)
			and nCOPD in (1, 2) 
			and nTypeOfCare <> 99 
			and nPnemonia in (1, 2) 
			and nICU <> 99 
			and nIntubated <> 99
			and dtSymptoms < '01-MAY-2020'
			and nAge >= @minAge
			and nAge <= @maxAge
			and (CASE nPregnancy WHEN 1 THEN '1' ELSE '0' END
			+ CASE nDiabetes WHEN 1 THEN '1' ELSE '0' END
			+ CASE nAthsma WHEN 1 THEN '1' ELSE '0' END
			+ CASE nImmunosuppressed WHEN 1 THEN '1' ELSE '0' END
			+ CASE nHypertension WHEN 1 THEN '1' ELSE '0' END
			+ CASE nOtherDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nCardiovascularDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nObesity WHEN 1 THEN '1' ELSE '0' END
			+ CASE nKidneyDisease WHEN 1 THEN '1' ELSE '0' END
			+ CASE nTobbacco WHEN 1 THEN '1' ELSE '0' END
			+ CASE nCOPD WHEN 1 THEN '1' ELSE '0' END) = @bindata
		GROUP BY CASE WHEN nAge < 10 THEN 5
					WHEN nAge < 20 THEN 15
					WHEN nAge < 30 THEN 25
					WHEN nAge < 40 THEN 35
					WHEN nAge < 50 THEN 45
					WHEN nAge < 60 THEN 55
					WHEN nAge < 70 THEN 65
					WHEN nAge < 80 THEN 75
					WHEN nAge < 90 THEN 85
					ELSE 95 END, CASE nSex WHEN 1 THEN 'Female' ELSE 'Male' END
		ORDER BY Age

		END
END


