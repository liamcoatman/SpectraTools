# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:50:19 2015

@author: lc585
"""

def makehtml():

    t = Table.read('/home/lc585/Dropbox/IoA/BlackHoleMasses/lineinfo_v3.dat',
                   format='ascii',
                   guess=False,
                   delimiter=',')

    links = []
    for n in t['Name']:

        objdir = os.path.join('/data/lc585/WHT_20150331/html',n)

        with open( os.path.join('/data/lc585/WHT_20150331/html/',n,'plots.html'),'w') as f:
            links.append(os.path.join('/data/lc585/WHT_20150331/html/',n,'plots.html'))
            lines = "<!DOCTYPE html> \n"
            lines += "<html> \n"
            lines += "<body> \n"
            lines += "<BR> \n"
            lines += "<h3>2D spectrum</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/2D_LR.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>New Spectrum</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_LR_Tell_BKGD_v138.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Without telluric correction</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_LR_l9.7.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>SDSS spectrum</h3> \n"
            lines += "<img src='{}/SDSS.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>SDSS+LIRIS (No Telluric Correction)</h3> \n"
            lines += "<img src='{}/1D_LR_v138_SDSS.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric Spectrum</h3> \n"
            lines += "<img src='{}/telluric.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric Spectrum / Blackbody</h3> \n"
            lines += "<img src='{}/ttelluric.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_LR_Tell_l9.7.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_LR_Tell_v138.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Fit to H-alpha</h3> \n"
            lines += "<img src='{}/HalphaFit.png' alt='No Fit' style='width:600px;height:800px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>CIV and Ha emission lines</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/CIV+Ha.png' alt='No Spectrum' style='width:800px;height:1000px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Hb emission lines</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/Hb.png' alt='No Spectrum' style='width:800px;height:1000px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>2D High-Resolution Spectra</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/2D_HR.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Without telluric correction</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_l2.6.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Without telluric correction - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_l2.6x2.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Without telluric correction - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_l2.6x3.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Without telluric correction - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_l2.6x4.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric Spectrum</h3> \n"
            lines += "<img src='{}/telluricHR.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric Spectrum / Blackbody</h3> \n"
            lines += "<img src='{}/ttelluricHR.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_Tell_l2.6.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_Tell_l2.6x2.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_Tell_l2.6x3.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "<BR> \n"
            lines += "<h3>Telluric corrected spectrum - weighted rebin</h3> \n"
            lines += "<BR> \n"
            lines += "<img src='{}/1D_HR_Tell_l2.6x4.png' alt='No Spectrum' style='width:800px;height:600px'> \n".format(objdir)
            lines += "</body> \n"
            lines += "</html> \n"
            f.write(lines)

    sedlinks = []
    for n in t['Name']:
        with open( os.path.join('/data/lc585/WHT_20150331/html/',n,'sedplots.html'),'w') as f:
            sedlinks.append(os.path.join('/data/lc585/WHT_20150331/html/',n,'sedplots.html'))
            f.write("<img src='sed.png' alt='No Spectrum' > ")


    lines = ""

    for j,row in enumerate(t):
        for i,e in enumerate(row):

            if i == 0:
                st = "<A HREF = " + links[j] + ">" + str(e) + "</A>"

            elif i == 1:
                st = "<A HREF = " + sedlinks[j] + ">" + str(e) + "</A>"

            else:
                st = str(e)

            if i == 0:
                line = "<TR><TD>" + st + "</TD>"

            elif i == len(row) - 1:
                line += "<TD>" + st + "</TD></TR>\n"

            else:
                line += "<TD>" + st + "</TD>"

        lines += line

    heads = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
    heads += "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
    heads += "<head>\n"
    heads += "<title>Liam's Table</title>\n"
    heads += "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\" />\n"
    heads += "<style type=\"text/css\" media=\"screen\">\n"
    heads += "body{margin:15px; padding:15px; border:1px solid #666; font-family:Arial, Helvetica, sans-serif; font-size:88%;}\n"
    heads += "h2{ margin-top: 50px; }\n"
    heads += "pre{ margin:5px; padding:5px; background-color:#f4f4f4; border:1px solid #ccc; }\n"
    heads += "th img{ border:0; }\n"
    heads += "th a{ color:#fff; font-size:13px; text-transform: uppercase; text-decoration:none; }\n"
    heads += "</style>\n"
    heads += "<script src=\"TableFilter/tablefilter_min.js\" language=\"javascript\" type=\"text/javascript\"></script>\n"
    heads += "<script src=\"TableFilter/TF_Modules/tf_paging.js\" language=\"javascript\" type=\"text/javascript\"></script>\n"
    heads += "<link rel=\"stylesheet\" type=\"text/css\" href=\"includes/SyntaxHighlighter/Styles/SyntaxHighlighter.css\" />\n"
    heads += "<script src=\"includes/SyntaxHighlighter/Scripts/shCore.js\" language=\"javascript\" type=\"text/javascript\"></script>\n"
    heads += "<script src=\"includes/SyntaxHighlighter/Scripts/shBrushJScript.js\" language=\"javascript\" type=\"text/javascript\"></script>\n"
    heads += "<script src=\"includes/SyntaxHighlighter/Scripts/shBrushXml.js\" language=\"javascript\" type=\"text/javascript\"></script>\n"
    heads += "<script language=\"javascript\" type=\"text/javascript\">\n"
    heads += "//<![CDATA[tf_AddEvent(window,\'load\',initHighlighter);function initHighlighter()\n"
    heads += "{dp.SyntaxHighlighter.ClipboardSwf = \"includes/SyntaxHighlighter/Scripts/clipboard.swf\"; dp.SyntaxHighlighter.HighlightAll(\"code\"); }\n"
    heads += "function hideIESelects(){if(tf_isIE){var slc = tf.tbl.getElementsByTagName(\'select\');for(var i=0; i<slc.length; i++)slc[i].style.visibility = \'hidden\';}}\n"
    heads += "function showIESelects(){if(tf_isIE){var slc = tf.tbl.getElementsByTagName(\'select\');for(var i=0; i<slc.length; i++)slc[i].style.visibility = \'visible\';}}//]]>\n"
    heads += "</script>\n"
    heads += "</head>\n"
    heads += "<body>\n"
    heads += "<h1>Results Table</h1>"
    heads += "<BR>"+"Info"+"<BR> \n"
    heads += "<table id=\"table1\" cellpadding=\"0\" cellspacing=\"0\">\n"
    heads += "<thead>\n"
    heads += "<tr>\n"

    for i,head in enumerate(t.colnames):
        heads += "<TH><B>" + head + "</B></TH>"

    heads += "\n</TR>\n</THEAD>\n<TBODY>\n"

    tails = "</div>\n"
    tails += "<div style=\"clear:both\"></div> \n"
    tails += "<script language=\"javascript\" type=\"text/javascript\">\n"
    tails += "//<![CDATA[\n"
    tails += "var props = {\n"
    tails += "filters_row_index: 1,\n"
    tails += "sort: true,\n"
    tails += "sort_config: {\n"
    tails += "sort_types:[\'US\',\'String\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\',\'US\']\n"
    tails += "},\n"
    tails += "col_38: 'select',\n"
    tails += "col_39: 'select',\n"
    tails += "col_42: 'select',\n"
    tails += "col_43: 'select',\n"
    tails += "col_44: 'select',\n"
    tails += "remember_grid_values: true,\n"
    tails += "alternate_rows: true,\n"
    tails += "paging: true,\n"
    tails += "results_per_page: [\'Results per page\',[10,25,50,100]],\n"
    tails += "rows_counter: true,\n"
    tails += "rows_counter_text: \"Displayed rows: \",\n"
    tails += "btn_reset: true,\n"
    tails += "btn_reset_text: \"Clear\",\n"
    tails += "btn_text: \" > \",\n"
    tails += "loader: true,\n"
    tails += "loader_text: \"Filtering data...\",\n"
    tails += "loader_html: \'<img src=\"loader.gif\" alt=\"\" \' +\n"
    tails += "\'style=\"vertical-align:middle;\" /> Loading...\',\n"
    tails += "on_show_loader: hideIESelects, //IE only: selects are hidden when loader visible\n"
    tails += "on_hide_loader: showIESelects, //IE only: selects are displayed when loader closed\n"
    tails += "display_all_text: \"< Show all >\",\n"
    tails += "}\n"
    tails += "var tf = setFilterGrid(\"table1\",props);\n"
    tails += "//]]>\n"
    tails += "</script>\n"
    tails += "</body>\n"
    tails += "</html>\n"

    lines = heads + lines + tails

    with open('/data/lc585/WHT_20150331/html/index.html', "w") as f:
        f.write(lines)

    t.write('/data/lc585/WHT_20150331/html/out.fits',overwrite=True)

    print 'Done!'

    return None

#if __name__ == "__main__":
#    makehtml()