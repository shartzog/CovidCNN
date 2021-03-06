/****** Object:  StoredProcedure [dbo].[getpydatav2]    Script Date: 6/19/2020 10:43:27 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Sam Hartzog
-- Create date: 5/20/20
-- Description:	Get data for python
-- =============================================
ALTER PROCEDURE [dbo].[getpydatav2]
AS
BEGIN
	SELECT CASE nSex WHEN 1 THEN '0' ELSE '1' END 
      + CASE nPregnancy WHEN 1 THEN '1' ELSE '0' END
      + CASE nDiabetes WHEN 1 THEN '1' ELSE '0' END
      + CASE nAthsma WHEN 1 THEN '1' ELSE '0' END
      + CASE nImmunosuppressed WHEN 1 THEN '1' ELSE '0' END
      + CASE nHypertension WHEN 1 THEN '1' ELSE '0' END
      + CASE nOtherDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nCardiovascularDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nObesity WHEN 1 THEN '1' ELSE '0' END
      + CASE nKidneyDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nTobbacco WHEN 1 THEN '1' ELSE '0' END
      + CASE nCOPD WHEN 1 THEN '1' ELSE '0' END
	  AS binData,
	  nAge as Age,
	  CASE nTypeOfCare WHEN 2 THEN '1' ELSE '0' END
      + CASE WHEN (nICU = 1 OR nIntubated = 1) THEN '1' ELSE '0' END
      + CASE dtDeath WHEN '1970-01-01 00:00:00.000' THEN '0' ELSE '1' END
      + CASE nPnemonia WHEN 1 THEN '1' ELSE '0' END
	  AS labels
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
END


