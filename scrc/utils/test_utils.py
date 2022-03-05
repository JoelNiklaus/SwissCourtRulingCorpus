import json
import pytest

from scrc.enums.language import Language
from scrc.enums.section import Section
import scrc.preprocessors.extractors.spider_specific.court_composition_extracting_functions as c
import scrc.preprocessors.extractors.spider_specific.procedural_participation_extracting_functions as p

"""
    Helper module to test the functions of other modules.
    Allows to check if the court composition extraction and the procedural participation extraction returns the expected result for a given test input.

    Classes:
        `TestCourtCompositionExtractingFunctions`
        `TestProceduralParticipationExtractingFunctions`

    Usage:
        Assuming you are in the SwissCourtRulingCorpus directory and use the command line:
        Run all tests with `pytest`
        Run only this module with `pytest scrc/utils/test_utils.py`
        Run a specific class with `pytest -k "ClassName"`
            e.g. `pytest -k "TestCourtCompositionExtractingFunctions"`
        Run a specific function with `pytest -k "ClassName and function_name"`
            e.g. `pytest -k "TestCourtCompositionExtractingFunctions and test_equality"`
        Consult `https://docs.pytest.org/en/6.2.x/usage.html` for more information.

    Adding tests:
        Add tests for a new module by creating a new class.
        Add tests for a spider by 
            - adding new test headers
            - adding a new setup line in the module that you want to test
            - adding what you want to test and what you expect to the test_data
        Add new kinds of test by adding new functions.
"""


ZG_Verwaltungsgericht_test_header = ['Normal.dot', 'VERWALTUNGSGERICHT DES KANTONS ZUG', 'SOZIALVERSICHERUNGSRECHTLICHE KAMMER', 'Mitwirkende Richter: lic. iur. Adrian Willimann, Vorsitz lic. iur. Jacqueline Iten-Staub und Dr. iur. Matthias Suter Gerichtsschreiber: MLaw Patrick Trütsch', 'U R T E I L vom 18. Juni 2020 [rechtskräftig] gemäss § 29 der Geschäftsordnung', 'in Sachen', 'A._ Beschwerdeführer vertreten durch B._ AG', 'gegen', 'Ausgleichskasse Zug, Baarerstrasse 11, Postfach, 6302 Zug Beschwerdegegnerin', 'betreffend', 'Ergänzungsleistungen (hypothetisches Erwerbseinkommen)', 'S 2019 121', '2', 'Urteil S 2019 121']

ZG_Verwaltungsgericht_test_header_2 = ['Normal.dot', 'VERWALTUNGSGERICHT DES KANTONS ZUG', 'SOZIALVERSICHERUNGSRECHTLICHE KAMMER', 'Mitwirkende Richter: lic. iur. Adrian Willimann, Vorsitz Dr. iur. Matthias Suter und MLaw Ines Stocker Gerichtsschreiber: MLaw Patrick Trütsch', 'U R T E I L vom 19. Oktober 2020 [rechtskräftig] gemäss § 29 der Geschäftsordnung', 'in Sachen', 'A._ Beschwerdeführer vertreten durch B._ AG', 'gegen', 'Amt für Wirtschaft und Arbeit (AWA), vertreten durch Arbeitslosenkasse des Kantons Zug, Rechtsdienst, Industriestrasse 24, 6301 Zug Beschwerdegegner', 'betreffend', 'Arbeitslosenversicherung (Einstellung in der Anspruchsberechtigung)', 'S 2020 12', '2', 'Urteil S 2020 12']

ZH_Steuerrekurs_test_header = ['Endentscheid Kammer', 'Steuerrekursgericht des Kantons Zürich', '2. Abteilung', '2 DB.2017.240 2 ST.2017.296', 'Entscheid', '5. Februar 2019', 'Mitwirkend:', 'Abteilungspräsident Christian Mäder, Steuerrichterin Micheline Roth, Steuerrichterin Barbara Collet und Gerichtsschreiber Hans Heinrich Knüsli', 'In Sachen', '1. A, 2. B,', 'Beschwerdeführer/ Rekurrenten, vertreten durch C AG,', 'gegen', '1. Schw eizer ische E idgenossenschaf t , Beschwerdegegnerin, 2. Staat Zür ich , Rekursgegner, vertreten durch das kant. Steueramt, Division Konsum, Bändliweg 21, Postfach, 8090 Zürich,', 'betreffend', 'Direkte Bundessteuer 2012 sowie Staats- und Gemeindesteuern 2012', '- 2 -', '2 DB.2017.240 2 ST.2017.296']

ZH_Steuerrekurs_test_header_2 = ['Endentscheid Kammer', 'Steuerrekursgericht des Kantons Zürich', '2. Abteilung', '2 GR.2013.5', 'Entscheid', '26. August 2013', 'Mitwirkend:', 'Abteilungspräsident Christian Mäder, Steuerrichter Alexander Widl, Ersatzrichter Claude Treyer und Gerichtsschreiber Stefan Eichenberger', 'In Sachen', 'A Gm bH, vormals B GmbH,', 'als Rechtsnachfolgerin der C GmbH Immobiliengesellschaft,', 'Rekurrentin, vertreten durch Ernst & Young AG, Maagplatz 1, Postfach, 8010 Zürich,', 'gegen', 'Gem einde D , Rekursgegnerin, vertreten durch die Kommission für Grundsteuern,', 'betreffend', 'Grundstückgewinnsteuer', '- 2 -', '2 GR.2013.5']

ZH_Baurekurs_test_header = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2018.00197 und R2.2019.00057 BRGE II Nr. 0142/2019 und 0143/2019', 'Entscheid vom 10. September 2019', 'Mitwirkende Abteilungsvizepräsident Adrian Bergmann, Baurichter Stefano Terzi,  Marlen Patt, Gerichtsschreiber Daniel Schweikert', 'in Sachen Rekurrentin', 'V. L. [...]', 'vertreten durch [...]', 'gegen Rekursgegnerschaft', '1. Baubehörde X 2. M. I. und K. I.-L. [...]', 'Nr. 2 vertreten durch [...]', 'R2.2018.00197 betreffend Baubehördenbeschluss vom 4. September 2017; Baubewilligung für Um-', 'bau Einfamilienhausteil und Ausbau Dachgeschoss, [...], BRGE II Nr. 00025/2018 vom 6. März 2018; Rückweisung zum  mit VB.2018.00209 vom 20. September 2018', 'R2.2019.00057 Präsidialverfügung vom 29. März 2019; Baubewilligung für Umbau  und Ausbau Dachgeschoss (1. Projektänderung), [...] _', 'R2.2018.00197 Seite 2']

ZH_Baurekurs_test_header_2 = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2011.00160 BRGE II Nr. 0049/2012', 'Entscheid vom 20. März 2012', 'Mitwirkende Abteilungsvizepräsident Emil Seliner, Baurichter Peter Rütimann,  Adrian Bergmann, Gerichtsschreiber Robert Durisch', 'in Sachen Rekurrentin', 'Hotel Uto Kulm AG, Gratstrasse, 8143 Stallikon', 'vertreten durch Rechtsanwalt Dr. iur. Christof Truniger, Metzgerrainle 9, Postfach 5024, 6000 Luzern 5', 'gegen Rekursgegnerinnen', '1. Bau- und Planungskommission Stallikon, 8143 Stallikon 2. Baudirektion Kanton Zürich, Walchetor, Walcheplatz 2, Postfach,', '8090 Zürich', 'betreffend Bau- und Planungskommissionsbeschluss vom 24. August 2011 und Ver-', 'fügung der Baudirektion Kanton Zürich Nr. BVV 06.0429_1 vom 8. Juli 2011; Verweigerung der nachträglichen Baubewilligung für Aussen- und Turmbeleuchtung Uto Kulm (Neubeurteilung), Kat.-Nr. 1032, Gratstrasse, Hotel-Restaurant Uto Kulm, Üetliberg / Stallikon _', 'R2.2011.00160 Seite 2']

ZH_Obergericht_test_header = ['Urteil - Abweisung, begründet', 'Bezirksgericht Zürich 3. Abteilung', 'Geschäfts-Nr.: CG170019-L / U', 'Mitwirkend: Vizepräsident lic. iur. Th. Kläusli, Bezirksrichter lic. iur. K. Vogel,', 'Ersatzrichter MLaw D. Brugger sowie der Gerichtsschreiber M.A.', 'HSG Ch. Reitze', 'Urteil vom 4. März 2020', 'in Sachen', 'A._, Kläger', 'vertreten durch Rechtsanwalt lic. iur. W._', 'gegen', '1. B._, 2. C._-Stiftung, 3. D._, Beklagte', '1 vertreten durch Rechtsanwalt Dr. iur. X._', '2 vertreten durch Rechtsanwältin Dr. iur. Y._']

ZH_Obergericht_test_header_2 = ['Kassationsgericht des Kantons Zürich', 'Kass.-Nr. AA050130/U/mb', 'Mitwirkende: die Kassationsrichter Moritz Kuhn, Präsident, Robert Karrer, Karl', 'Spühler, Paul Baumgartner und die Kassationsrichterin Yvona', 'Griesser sowie die Sekretärin Margrit Scheuber', 'Zirkulationsbeschluss vom 4. September 2006', 'in Sachen', 'A. X., geboren ..., von ..., whft. in ...,', 'Klägerin, Rekurrentin, Anschlussrekursgegnerin und Beschwerdeführerin vertreten durch Rechtsanwalt Dr. iur. C. D.', 'gegen', 'B. X., geboren ..., von ..., whft. in ...,', 'Beklagter, Rekursgegner, Anschlussrekurrent und Beschwerdegegner vertreten durch Rechtsanwältin lic. iur. E. F.']

ZH_Obergericht_test_header_3 = ['Urteil Gutheissung/Abweisung Beschwerde', 'Obergericht des Kantons Zürich I. Zivilkammer', 'Geschäfts-Nr.: PP110014-O/U', 'Mitwirkend: Oberrichter Dr. R. Klopfer, Vorsitzender, Oberrichterin Dr. M. Schaffitz', 'und Oberrichter lic. iur. M. Spahn sowie Gerichtsschreiberin lic. iur.', 'C. Heuberger', 'Urteil vom 28. September 2011', 'in Sachen', 'A._, Beklagte und Beschwerdeführerin', 'gegen', 'B._, Kläger und Beschwerdegegner', 'vertreten durch Rechtsanwältin mag. iur. et lic. oec. publ. X._']

ZH_Verwaltungsgericht_test_header = ['Verwaltungsgericht des Kantons Zürich 4. Abteilung', 'VB.2020.00452', 'Urteil', 'der 4. Kammer', 'vom 24. September 2020', 'Mitwirkend: Abteilungspräsidentin Tamara Nüssle (Vorsitz), Verwaltungsrichter Reto Häggi Furrer, Verwaltungsrichter Martin Bertschi, Gerichtsschreiber David Henseler.', 'In Sachen', 'A, vertreten durch RA B,', 'Beschwerdeführerin,', 'gegen', 'Migrationsamt des Kantons Zürich,', 'Beschwerdegegner,', 'betreffend vorzeitige Erteilung der Niederlassungsbewilligung,']

ZH_Verwaltungsgericht_test_header_2 = ['Verwaltungsgericht des Kantons Zürich 3. Abteilung', 'VB.2011.00558', 'Urteil', 'der 3. Kammer', 'vom 8. Februar 2012', 'Mitwirkend: Abteilungspräsident Rudolf Bodmer (Vorsitz), Verwaltungsrichterin Bea Rotach Tomschin, Ersatzrichter Martin Kayser, Gerichtsschreiber Cyrill Bienz.', 'In Sachen', 'Stadt Zürich, vertreten durch das Polizeidepartement,', 'Beschwerdeführerin,', 'gegen', 'A, vertreten durch RA B,', 'Beschwerdegegner,', 'betreffend Benützung des öffentlichen Grundes zu Sonderzwecken,']


ZH_Sozialversicherungsgericht_test_header = ['Sozialversicherungsgerichtdes Kantons Zürich IV.2014.00602', 'II. Kammer', 'Sozialversicherungsrichter Mosimann, Vorsitzender', 'Sozialversicherungsrichterin Käch', 'Sozialversicherungsrichterin Sager', 'Gerichtsschreiberin Kudelski', 'Urteil vom 11. August 2015', 'in Sachen', 'X._', 'Beschwerdeführerin', 'vertreten durch Rechtsanwalt Dr. Kreso Glavas', 'Advokatur Glavas AG', 'Markusstrasse 10, 8006 Zürich', 'gegen', 'Sozialversicherungsanstalt des Kantons Zürich, IV-Stelle', 'Röntgenstrasse 17, Postfach, 8087 Zürich', 'Beschwerdegegnerin', 'weitere Verfahrensbeteiligte:', 'Personalvorsorgestiftung der Y._', 'Beigeladene']

ZH_Sozialversicherungsgericht_test_header_2 = ['BV.2008.00114', 'Sozialversicherungsgericht', 'des Kantons Zürich', 'III. Kammer', 'Sozialversicherungsrichterin Heine, Vorsitzende', 'Sozialversicherungsrichterin Annaheim', 'Sozialversicherungsrichterin Daubenmeyer', 'Gerichtssekretär O. Peter', 'Urteil vom 30. Juni 2010', 'in Sachen', 'X._', 'Klägerin', 'vertreten durch Rechtsdienst Integration Handicap', 'Bürglistrasse 11, 8002 Zürich', 'gegen', 'GastroSocial Pensionskasse', 'Bahnhofstrasse 86, Postfach, 5001 Aarau', 'Beklagte', 'vertreten durch Rechtsanwältin Dr. Isabelle Vetter-Schreiber', 'Hubatka Müller & Vetter, Rechtsanwälte', 'Seestrasse 6, Postfach 1544, 8027 Zürich']

ZH_Sozialversicherungsgericht_test_header_3 = ['Sozialversicherungsgerichtdes Kantons Zürich', 'IV.2017.00330 IV. Kammer Sozialversicherungsrichter Hurst, Vorsitzender Sozialversicherungsrichterin Philipp Sozialversicherungsrichter Vogel Gerichtsschreiberin Curiger Urteil vom 7. August 2018', 'in Sachen', 'X._', 'Beschwerdeführerin', 'vertreten durch Rechtsanwalt Christoph Erdös', 'Erdös & Lehmann Rechtsanwälte', 'Kernstrasse 37, 8004 Zürich', 'gegen', 'Sozialversicherungsanstalt des Kantons Zürich, IV-Stelle', 'Röntgenstrasse 17, Postfach, 8087 Zürich', 'Beschwerdegegnerin']

namespace_de = {'language': Language.DE}
namespace_fr = {'language': Language.FR}
namespace_it = {'language': Language.IT}




class TestCourtCompositionExtractingFunctions:
    """
    This class tests whether the court composition extracting functions give the correct procedural participation for a given input. If the output is incorrect, an error is shown. 
    """

    def court_composition_setup(court, test_list, namespace):
        """
        :param court:       the court to be tested
        :param test_list:   the header to be tested
        :param namespace:   the header's language
        :return:            the CourtComposition
        """
        test_string = ' '.join(map(str, test_list))
        sections = {}
        sections[Section.HEADER] = test_string
        return court(sections, namespace)
        
    zg_vg = court_composition_setup(c.ZG_Verwaltungsgericht, ZG_Verwaltungsgericht_test_header, namespace_de)
    zg_vg_2 = court_composition_setup(c.ZG_Verwaltungsgericht, ZG_Verwaltungsgericht_test_header_2, namespace_de)
    zh_sr = court_composition_setup(c.ZH_Steuerrekurs, ZH_Steuerrekurs_test_header, namespace_de)
    zh_sr_2 = court_composition_setup(c.ZH_Steuerrekurs, ZH_Steuerrekurs_test_header_2, namespace_de)
    zh_br = court_composition_setup(c.ZH_Baurekurs, ZH_Baurekurs_test_header, namespace_de)
    zh_br_2 = court_composition_setup(c.ZH_Baurekurs, ZH_Baurekurs_test_header_2, namespace_de)
    zh_og = court_composition_setup(c.ZH_Obergericht, ZH_Obergericht_test_header, namespace_de)
    zh_og_2 = court_composition_setup(c.ZH_Obergericht, ZH_Obergericht_test_header_2, namespace_de)
    zh_vg = court_composition_setup(c.ZH_Verwaltungsgericht, ZH_Verwaltungsgericht_test_header, namespace_de)
    zh_vg_2 = court_composition_setup(c.ZH_Verwaltungsgericht, ZH_Verwaltungsgericht_test_header_2, namespace_de)
    zh_svg = court_composition_setup(c.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header, namespace_de)
    zh_svg_2 = court_composition_setup(c.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header_2, namespace_de)
    zh_svg_3 = court_composition_setup(c.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header_3, namespace_de)

    test_data = [
        # The first element of each tuple is the value to be tested,
        # and the second element is the expected result.
        # ZG_Verwaltungsgericht
        (zg_vg.president.name, 'Adrian Willimann'),
        (zg_vg.judges[0].name, 'Adrian Willimann'), 
        (zg_vg.judges[1].name, 'Jacqueline Iten-Staub'),
        (zg_vg.judges[2].name, 'Matthias Suter'),
        (zg_vg.clerks[0].name, 'Patrick Trütsch'),
        # ZG_Verwaltungsgericht
        (zg_vg_2.president.name, 'Adrian Willimann'),
        (zg_vg_2.judges[0].name, 'Adrian Willimann'), 
        (zg_vg_2.judges[1].name, 'Matthias Suter'),
        (zg_vg_2.judges[2].name, 'Ines Stocker'),
        (zg_vg_2.clerks[0].name, 'Patrick Trütsch'),
        # ZH_Steuerrekurs
        (zh_sr.president.name, 'Christian Mäder'),
        (zh_sr.president.gender.value, 'male'),
        (zh_sr.judges[0].name, 'Christian Mäder'),
        (zh_sr.judges[0].gender.value, 'male'),
        (zh_sr.judges[1].name, 'Micheline Roth'),
        (zh_sr.judges[1].gender.value, 'female'),
        (zh_sr.judges[2].name, 'Barbara Collet'),
        (zh_sr.judges[2].gender.value, 'female'),
        (zh_sr.clerks[0].name, 'Hans Heinrich Knüsli'),
        (zh_sr.clerks[0].gender.value, 'male'),
        # ZH_Steuerrekurs
        (zh_sr_2.president.name, 'Christian Mäder'),
        (zh_sr_2.president.gender.value, 'male'),
        (zh_sr_2.judges[0].name, 'Christian Mäder'),
        (zh_sr_2.judges[0].gender.value, 'male'),
        (zh_sr_2.judges[1].name, 'Alexander Widl'),
        (zh_sr_2.judges[1].gender.value, 'male'),
        (zh_sr_2.judges[2].name, 'Claude Treyer'),
        (zh_sr_2.judges[2].gender.value, 'male'),
        (zh_sr_2.clerks[0].name, 'Stefan Eichenberger'),
        (zh_sr_2.clerks[0].gender.value, 'male'),
        # ZH_Baurekurs
        (zh_br.judges[0].name, 'Adrian Bergmann'),
        (zh_br.judges[0].gender.value, 'male'),
        (zh_br.judges[1].name, 'Stefano Terzi'),
        (zh_br.judges[1].gender.value, 'male'),
        (zh_br.judges[2].name, 'Marlen Patt'),
        (zh_br.judges[2].gender.value, 'male'),
        (zh_br.clerks[0].name, 'Daniel Schweikert'),
        (zh_br.clerks[0].gender.value, 'male'),
        # ZH_Baurekurs
        (zh_br_2.judges[0].name, 'Emil Seliner'),
        (zh_br_2.judges[0].gender.value, 'male'),
        (zh_br_2.judges[1].name, 'Peter Rütimann'),
        (zh_br_2.judges[1].gender.value, 'male'),
        (zh_br_2.judges[2].name, 'Adrian Bergmann'),
        (zh_br_2.judges[2].gender.value, 'male'),
        (zh_br_2.clerks[0].name, 'Robert Durisch'),
        (zh_br_2.clerks[0].gender.value, 'male'),
        # ZH_Obergericht
        (zh_og.judges[0].name, 'Th. Kläusli'),
        (zh_og.judges[0].gender.value, 'male'),
        (zh_og.judges[1].name, 'K. Vogel'),
        (zh_og.judges[1].gender.value, 'male'),
        (zh_og.judges[2].name, 'D. Brugger'),
        (zh_og.judges[2].gender.value, 'male'),
        (zh_og.clerks[0].name, 'Ch. Reitze'),
        (zh_og.clerks[0].gender.value, 'male'),
        # ZH_Obergericht
        (zh_og_2.president.name, 'Moritz Kuhn'),
        (zh_og_2.president.gender.value, 'male'),
        (zh_og_2.judges[0].name, 'Moritz Kuhn'),
        (zh_og_2.judges[0].gender.value, 'male'),
        (zh_og_2.judges[1].name, 'Robert Karrer'),
        (zh_og_2.judges[1].gender.value, 'male'),
        (zh_og_2.judges[2].name, 'Karl Spühler'),
        (zh_og_2.judges[2].gender.value, 'male'),
        (zh_og_2.judges[3].name, 'Paul Baumgartner'),
        (zh_og_2.judges[3].gender.value, 'male'),
        (zh_og_2.judges[4].name, 'Yvona Griesser'),
        (zh_og_2.judges[4].gender.value, 'female'),
        (zh_og_2.clerks[0].name, 'Margrit Scheuber'),
        (zh_og_2.clerks[0].gender.value, 'female'),
        # ZH_Verwaltungsgericht
        (zh_vg.president.name, 'Tamara Nüssle'),
        (zh_vg.president.gender.value, 'female'),
        (zh_vg.judges[0].name, 'Tamara Nüssle'),
        (zh_vg.judges[0].gender.value, 'female'),
        (zh_vg.judges[1].name, 'Reto Häggi Furrer'),
        (zh_vg.judges[1].gender.value, 'male'),
        (zh_vg.judges[2].name, 'Martin Bertschi'),
        (zh_vg.judges[2].gender.value, 'male'),
        (zh_vg.clerks[0].name, 'David Henseler'),
        (zh_vg.clerks[0].gender.value, 'male'),
        # ZH_Verwaltungsgericht
        (zh_vg_2.president.name, 'Rudolf Bodmer'),
        (zh_vg_2.president.gender.value, 'male'),
        (zh_vg_2.judges[0].name, 'Rudolf Bodmer'),
        (zh_vg_2.judges[0].gender.value, 'male'),
        (zh_vg_2.judges[1].name, 'Bea Rotach Tomschin'),
        (zh_vg_2.judges[1].gender.value, 'female'),
        (zh_vg_2.judges[2].name, 'Martin Kayser'),
        (zh_vg_2.judges[2].gender.value, 'male'),
        (zh_vg_2.clerks[0].name, 'Cyrill Bienz'),
        (zh_vg_2.clerks[0].gender.value, 'male'),
        # ZH_Sozialversicherungsgericht
        (zh_svg.president.name, 'Mosimann'),
        (zh_svg.president.gender.value, 'male'),
        (zh_svg.judges[0].name, 'Mosimann'),
        (zh_svg.judges[0].gender.value, 'male'),
        (zh_svg.judges[1].name, 'Käch'),
        (zh_svg.judges[1].gender.value, 'female'),
        (zh_svg.judges[2].name, 'Sager'),
        (zh_svg.judges[2].gender.value, 'female'),
        (zh_svg.clerks[0].name, 'Kudelski'),
        (zh_svg.clerks[0].gender.value, 'female'),
        # ZH_Sozialversicherungsgericht
        (zh_svg_2.president.name, 'Heine'),
        (zh_svg_2.president.gender.value, 'female'),
        (zh_svg_2.judges[0].name, 'Heine'),
        (zh_svg_2.judges[0].gender.value, 'female'),
        (zh_svg_2.judges[1].name, 'Annaheim'),
        (zh_svg_2.judges[1].gender.value, 'female'),
        (zh_svg_2.judges[2].name, 'Daubenmeyer'),
        (zh_svg_2.judges[2].gender.value, 'female'),
        (zh_svg_2.clerks[0].name, 'O. Peter'),
        (zh_svg_2.clerks[0].gender.value, 'male'),
        # ZH_Sozialversicherungsgericht
        (zh_svg_3.president.name, 'Hurst'),
        (zh_svg_3.president.gender.value, 'male'),
        (zh_svg_3.judges[0].name, 'Hurst'),
        (zh_svg_3.judges[0].gender.value, 'male'),
        (zh_svg_3.judges[1].name, 'Philipp'),
        (zh_svg_3.judges[1].gender.value, 'female'),
        (zh_svg_3.judges[2].name, 'Vogel'),
        (zh_svg_3.judges[2].gender.value, 'male'),
        (zh_svg_3.clerks[0].name, 'Curiger'),
        (zh_svg_3.clerks[0].gender.value, 'female'),
    ]

    @pytest.mark.parametrize("input, expected", test_data)
    def test_equality(self, input, expected):
        assert input == expected




class TestProceduralParticipationExtractingFunctions():
    """
    This class tests whether the procedural participation extracting functions give the correct procedural participation for a given input. If the output is incorrect, an error is shown. 
    """

    def procedural_participation_setup(court, test_list, namespace):
        """
        :param court:       the court to be tested
        :param test_list:   the header to be tested
        :param namespace:   the header's language
        :return:            the ProceduralParticipation
        """
        test_string = ', '.join(map(str, test_list))
        sections = {}
        sections[Section.HEADER] = test_string
        result = court(sections, namespace)
        return json.loads(result)

    zg_vg = procedural_participation_setup(p.ZG_Verwaltungsgericht, ZG_Verwaltungsgericht_test_header, namespace_de)
    zg_vg_2 = procedural_participation_setup(p.ZG_Verwaltungsgericht, ZG_Verwaltungsgericht_test_header_2, namespace_de)
    zh_sr = procedural_participation_setup(p.ZH_Steuerrekurs, ZH_Steuerrekurs_test_header, namespace_de)
    zh_sr_2 = procedural_participation_setup(p.ZH_Steuerrekurs, ZH_Steuerrekurs_test_header_2, namespace_de)
    zh_br = procedural_participation_setup(p.ZH_Baurekurs, ZH_Baurekurs_test_header, namespace_de)
    zh_br_2 = procedural_participation_setup(p.ZH_Baurekurs, ZH_Baurekurs_test_header_2, namespace_de)
    zh_og = procedural_participation_setup(p.ZH_Obergericht, ZH_Obergericht_test_header, namespace_de)
    zh_og_2 = procedural_participation_setup(p.ZH_Obergericht, ZH_Obergericht_test_header_2, namespace_de)
    zh_og_3 = procedural_participation_setup(p.ZH_Obergericht, ZH_Obergericht_test_header_3, namespace_de)
    zh_vg = procedural_participation_setup(p.ZH_Verwaltungsgericht, ZH_Verwaltungsgericht_test_header, namespace_de)
    zh_vg_2 = procedural_participation_setup(p.ZH_Verwaltungsgericht, ZH_Verwaltungsgericht_test_header_2, namespace_de)
    zh_svg = procedural_participation_setup(p.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header, namespace_de)
    zh_svg_2 = procedural_participation_setup(p.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header_2, namespace_de)
    zh_svg_3 = procedural_participation_setup(p.ZH_Sozialversicherungsgericht, ZH_Sozialversicherungsgericht_test_header_3, namespace_de)

    test_data = [
        # The first element of each tuple is the value to be tested,
        # and the second element is the expected result.
        # ZG_Verwaltungsgericht
        (zg_vg['plaintiffs'][0]['legal_counsel'][0]['name'], 'B._ AG'),
        (zg_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zg_vg['defendants'][0]['legal_counsel'], []),
        # ZG_Verwaltungsgericht
        (zg_vg_2['defendants'][0]['legal_counsel'][0]['name'], 'Arbeitslosenkasse des Kantons Zug'),
        (zg_vg_2['defendants'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zg_vg_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'B._ AG'),
        (zg_vg_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        # ZH_Steuerrekurs
        (zh_sr['defendants'][0]['legal_counsel'][0]['name'], 'Steueramt'),
        (zh_sr['defendants'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zh_sr['plaintiffs'][0]['legal_counsel'][0]['name'], 'C AG'),
        (zh_sr['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        # ZH_Steuerrekurs
        (zh_sr_2['defendants'][0]['legal_counsel'][0]['name'], 'Kommission für Grundsteuern'),
        (zh_sr_2['defendants'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zh_sr_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'Ernst & Young AG'),
        (zh_sr_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        # ZH_Baurekurs
        (zh_br['plaintiffs'][0]['legal_counsel'], []),
        (zh_br['defendants'][0]['legal_counsel'], []),
        # ZH_Baurekurs
        (zh_br_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'Christof Truniger'),
        (zh_br_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_br_2['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male'),
        (zh_br_2['defendants'][0]['legal_counsel'], []),
        # ZH_Obergericht
        (zh_og['plaintiffs'][0]['legal_counsel'][0]['name'], 'W._'),
        (zh_og['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_og['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male'),
        # ZH_Obergericht
        (zh_og_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'C. D.'),
        (zh_og_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_og_2['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male'),
        (zh_og_2['defendants'][0]['legal_counsel'][0]['name'], 'E. F.'),
        (zh_og_2['defendants'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_og_2['defendants'][0]['legal_counsel'][0]['gender'], 'female'),
        # ZH_Obergericht
        (zh_og_3['plaintiffs'][0]['legal_counsel'], []),
        (zh_og_3['defendants'][0]['legal_counsel'][0]['name'], 'X._'),
        (zh_og_3['defendants'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_og_3['defendants'][0]['legal_counsel'][0]['gender'], 'female'),
        # ZH_Verwaltungsgericht
        (zh_vg['plaintiffs'][0]['legal_counsel'][0]['name'], 'B'),
        (zh_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_vg['plaintiffs'][0]['legal_counsel'][0]['gender'], 'unknown'),
        (zh_vg['defendants'][0]['legal_counsel'], []),
        # ZH_Verwaltungsgericht
        (zh_vg_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'Polizeidepartement'),
        (zh_vg_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zh_vg_2['defendants'][0]['legal_counsel'][0]['name'], 'B'),
        (zh_vg_2['defendants'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_vg_2['defendants'][0]['legal_counsel'][0]['gender'], 'unknown'),
        # ZH_Sozialversicherungsgericht
        (zh_svg['plaintiffs'][0]['legal_counsel'][0]['name'], 'Kreso Glavas'),
        (zh_svg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_svg['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male'),
        (zh_svg['plaintiffs'][0]['legal_counsel'][0]['titles'][0], 'Dr.'),
        (zh_svg['defendants'][0]['legal_counsel'], []),
        # ZH_Sozialversicherungsgericht
        (zh_svg_2['plaintiffs'][0]['legal_counsel'][0]['name'], 'Rechtsdienst Integration Handicap'),
        (zh_svg_2['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity'),
        (zh_svg_2['defendants'][0]['legal_counsel'][0]['name'], 'Isabelle Vetter-Schreiber'),
        (zh_svg_2['defendants'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_svg_2['defendants'][0]['legal_counsel'][0]['gender'], 'female'),
        (zh_svg_2['defendants'][0]['legal_counsel'][0]['titles'][0], 'Dr.'),
        # ZH_Sozialversicherungsgericht
        (zh_svg_3['plaintiffs'][0]['legal_counsel'][0]['name'], 'Christoph Erdös'),
        (zh_svg_3['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person'),
        (zh_svg_3['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male'),
        (zh_svg_3['plaintiffs'][0]['legal_counsel'][0]['titles'], []),
        (zh_svg_3['defendants'][0]['legal_counsel'], []),
    ]

    @pytest.mark.parametrize("input, expected", test_data)
    def test_equality(self, input, expected):
        assert input == expected

