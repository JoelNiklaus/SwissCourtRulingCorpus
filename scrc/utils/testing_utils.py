import json
import unittest

from scrc.enums.language import Language
from scrc.enums.section import Section
import scrc.preprocessors.extractors.spider_specific.court_composition_extracting_functions as c
import scrc.preprocessors.extractors.spider_specific.procedural_participation_extracting_functions as p

"""
    Helper module to test the functions of other modules.
    Allows to check if the court composition extraction and the procedural participation extraction returns the expected result for a given test input.

    Classes:
        `test_court_composition_extracting_functions()`
        `test_procedural_participation_extracting_functions()`

    Usage:
        Run from the command line with `python -m unittest scrc.utils.testing_utils`
        Run with `python -m unittest -v scrc.utils.testing_utils` to get more verbose output
        Run specific classes with e.g. `python -m unittest scrc.utils.testing_utils.test_court_composition_extracting_functions`
        Run specific functions with e.g. `python -m unittest scrc.utils.testing_utils.test_court_composition_extracting_functions.test_ZG_Verwaltungsgericht`

    Adding tests:
        Add tests for a new module by creating a new class.
        Add tests for a new spider by creating a new function in the corresponding class.
        Add more tests for an existing spider by including more assertions in the corresponding function.

"""


ZG_Verwaltungsgericht_test_header = ['Normal.dot', 'VERWALTUNGSGERICHT DES KANTONS ZUG', 'SOZIALVERSICHERUNGSRECHTLICHE KAMMER', 'Mitwirkende Richter: lic. iur. Adrian Willimann, Vorsitz lic. iur. Jacqueline Iten-Staub und Dr. iur. Matthias Suter Gerichtsschreiber: MLaw Patrick Trütsch', 'U R T E I L vom 18. Juni 2020 [rechtskräftig] gemäss § 29 der Geschäftsordnung', 'in Sachen', 'A._ Beschwerdeführer vertreten durch B._ AG', 'gegen', 'Ausgleichskasse Zug, Baarerstrasse 11, Postfach, 6302 Zug Beschwerdegegnerin', 'betreffend', 'Ergänzungsleistungen (hypothetisches Erwerbseinkommen)', 'S 2019 121', '2', 'Urteil S 2019 121']

ZH_Steuerrekurs_test_header = ['Endentscheid Kammer', 'Steuerrekursgericht des Kantons Zürich', '2. Abteilung', '2 DB.2017.240 2 ST.2017.296', 'Entscheid', '5. Februar 2019', 'Mitwirkend:', 'Abteilungspräsident Christian Mäder, Steuerrichterin Micheline Roth, Steuerrichterin Barbara Collet und Gerichtsschreiber Hans Heinrich Knüsli', 'In Sachen', '1. A, 2. B,', 'Beschwerdeführer/ Rekurrenten, vertreten durch C AG,', 'gegen', '1. Schw eizer ische E idgenossenschaf t , Beschwerdegegnerin, 2. Staat Zür ich , Rekursgegner, vertreten durch das kant. Steueramt, Division Konsum, Bändliweg 21, Postfach, 8090 Zürich,', 'betreffend', 'Direkte Bundessteuer 2012 sowie Staats- und Gemeindesteuern 2012', '- 2 -', '2 DB.2017.240 2 ST.2017.296']

ZH_Baurekurs_test_header = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2018.00197 und R2.2019.00057 BRGE II Nr. 0142/2019 und 0143/2019', 'Entscheid vom 10. September 2019', 'Mitwirkende Abteilungsvizepräsident Adrian Bergmann, Baurichter Stefano Terzi,  Marlen Patt, Gerichtsschreiber Daniel Schweikert', 'in Sachen Rekurrentin', 'V. L. [...]', 'vertreten durch [...]', 'gegen Rekursgegnerschaft', '1. Baubehörde X 2. M. I. und K. I.-L. [...]', 'Nr. 2 vertreten durch [...]', 'R2.2018.00197 betreffend Baubehördenbeschluss vom 4. September 2017; Baubewilligung für Um-', 'bau Einfamilienhausteil und Ausbau Dachgeschoss, [...], BRGE II Nr. 00025/2018 vom 6. März 2018; Rückweisung zum  mit VB.2018.00209 vom 20. September 2018', 'R2.2019.00057 Präsidialverfügung vom 29. März 2019; Baubewilligung für Umbau  und Ausbau Dachgeschoss (1. Projektänderung), [...] _', 'R2.2018.00197 Seite 2']

ZH_Baurekurs_test_header_2 = ['BRGE Nr. 0/; GUTH vom', 'Baurekursgericht des Kantons Zürich', '2. Abteilung', 'G.-Nr. R2.2011.00160 BRGE II Nr. 0049/2012', 'Entscheid vom 20. März 2012', 'Mitwirkende Abteilungsvizepräsident Emil Seliner, Baurichter Peter Rütimann,  Adrian Bergmann, Gerichtsschreiber Robert Durisch', 'in Sachen Rekurrentin', 'Hotel Uto Kulm AG, Gratstrasse, 8143 Stallikon', 'vertreten durch Rechtsanwalt Dr. iur. Christof Truniger, Metzgerrainle 9, Postfach 5024, 6000 Luzern 5', 'gegen Rekursgegnerinnen', '1. Bau- und Planungskommission Stallikon, 8143 Stallikon 2. Baudirektion Kanton Zürich, Walchetor, Walcheplatz 2, Postfach,', '8090 Zürich', 'betreffend Bau- und Planungskommissionsbeschluss vom 24. August 2011 und Ver-', 'fügung der Baudirektion Kanton Zürich Nr. BVV 06.0429_1 vom 8. Juli 2011; Verweigerung der nachträglichen Baubewilligung für Aussen- und Turmbeleuchtung Uto Kulm (Neubeurteilung), Kat.-Nr. 1032, Gratstrasse, Hotel-Restaurant Uto Kulm, Üetliberg / Stallikon _', 'R2.2011.00160 Seite 2']

ZH_Obergericht_test_header = ['Urteil - Abweisung, begründet', 'Bezirksgericht Zürich 3. Abteilung', 'Geschäfts-Nr.: CG170019-L / U', 'Mitwirkend: Vizepräsident lic. iur. Th. Kläusli, Bezirksrichter lic. iur. K. Vogel,', 'Ersatzrichter MLaw D. Brugger sowie der Gerichtsschreiber M.A.', 'HSG Ch. Reitze', 'Urteil vom 4. März 2020', 'in Sachen', 'A._, Kläger', 'vertreten durch Rechtsanwalt lic. iur. W._', 'gegen', '1. B._, 2. C._-Stiftung, 3. D._, Beklagte', '1 vertreten durch Rechtsanwalt Dr. iur. X._', '2 vertreten durch Rechtsanwältin Dr. iur. Y._']

ZH_Obergericht_test_header_2 = ['Kassationsgericht des Kantons Zürich', 'Kass.-Nr. AA050130/U/mb', 'Mitwirkende: die Kassationsrichter Moritz Kuhn, Präsident, Robert Karrer, Karl', 'Spühler, Paul Baumgartner und die Kassationsrichterin Yvona', 'Griesser sowie die Sekretärin Margrit Scheuber', 'Zirkulationsbeschluss vom 4. September 2006', 'in Sachen', 'A. X., geboren ..., von ..., whft. in ...,', 'Klägerin, Rekurrentin, Anschlussrekursgegnerin und Beschwerdeführerin vertreten durch Rechtsanwalt Dr. iur. C. D.', 'gegen', 'B. X., geboren ..., von ..., whft. in ...,', 'Beklagter, Rekursgegner, Anschlussrekurrent und Beschwerdegegner vertreten durch Rechtsanwältin lic. iur. E. F.']

ZH_Verwaltungsgericht_test_header = ['Verwaltungsgericht des Kantons Zürich 4. Abteilung', 'VB.2020.00452', 'Urteil', 'der 4. Kammer', 'vom 24. September 2020', 'Mitwirkend: Abteilungspräsidentin Tamara Nüssle (Vorsitz), Verwaltungsrichter Reto Häggi Furrer, Verwaltungsrichter Martin Bertschi, Gerichtsschreiber David Henseler.', 'In Sachen', 'A, vertreten durch RA B,', 'Beschwerdeführerin,', 'gegen', 'Migrationsamt des Kantons Zürich,', 'Beschwerdegegner,', 'betreffend vorzeitige Erteilung der Niederlassungsbewilligung,']

ZH_Sozialversicherungsgericht_test_header = ['Sozialversicherungsgerichtdes Kantons ZürichIV.2014.00602', 'II. Kammer', 'Sozialversicherungsrichter Mosimann, Vorsitzender', 'Sozialversicherungsrichterin Käch', 'Sozialversicherungsrichterin Sager', 'Gerichtsschreiberin Kudelski', 'Urteil vom 11. August 2015', 'in Sachen', 'X._', 'Beschwerdeführerin', 'vertreten durch Rechtsanwalt Dr. Kreso Glavas', 'Advokatur Glavas AG', 'Markusstrasse 10, 8006 Zürich', 'gegen', 'Sozialversicherungsanstalt des Kantons Zürich, IV-Stelle', 'Röntgenstrasse 17, Postfach, 8087 Zürich', 'Beschwerdegegnerin', 'weitere Verfahrensbeteiligte:', 'Personalvorsorgestiftung der Y._', 'Beigeladene']

namespace_de = {'language': Language.DE}

class test_court_composition_extracting_functions(unittest.TestCase):
    """
    This function tests whether the court composition extracting functions give the correct procedural participation for a given input. If the output is incorrect, an error is shown. 
    """

    def test_ZG_Verwaltungsgericht(self):
        ZG_Verwaltungsgericht_test_string = ' '.join(map(str, ZG_Verwaltungsgericht_test_header))
        sections = {}
        sections[Section.HEADER] = ZG_Verwaltungsgericht_test_string
        zg_vg = c.ZG_Verwaltungsgericht(sections, namespace_de)
        # No tests for the gender because this court uses a generic masculine noun for multiple judges
        self.assertEqual(zg_vg.president.name, 'Adrian Willimann')
        self.assertEqual(zg_vg.judges[0].name, 'Adrian Willimann')
        self.assertEqual(zg_vg.judges[1].name, 'Jacqueline Iten-Staub')
        self.assertEqual(zg_vg.judges[2].name, 'Matthias Suter')
        self.assertEqual(zg_vg.clerks[0].name, 'Patrick Trütsch')

    def test_ZH_Steuerrekurs(self):
        ZH_Steuerrekurs_test_string = ' '.join(map(str, ZH_Steuerrekurs_test_header))
        sections = {}
        sections[Section.HEADER] = ZH_Steuerrekurs_test_string
        zh_sr = c.ZH_Steuerrekurs(sections, namespace_de)
        self.assertEqual(zh_sr.president.name, 'Christian Mäder')
        self.assertEqual(zh_sr.president.gender.value, 'male')
        self.assertEqual(zh_sr.judges[0].name, 'Christian Mäder')
        self.assertEqual(zh_sr.judges[0].gender.value, 'male')
        self.assertEqual(zh_sr.judges[1].name, 'Micheline Roth')
        self.assertEqual(zh_sr.judges[1].gender.value, 'female')
        self.assertEqual(zh_sr.judges[2].name, 'Barbara Collet')
        self.assertEqual(zh_sr.judges[2].gender.value, 'female')
        self.assertEqual(zh_sr.clerks[0].name, 'Hans Heinrich Knüsli')
        self.assertEqual(zh_sr.clerks[0].gender.value, 'male')

    def test_ZH_Baurekurs(self):
        ZH_Baurekurs_test_string = ' '.join(map(str, ZH_Baurekurs_test_header))
        sections = {}
        sections[Section.HEADER] = ZH_Baurekurs_test_string
        zh_br = c.ZH_Baurekurs(sections, namespace_de) 
        self.assertIsNone(zh_br.president)
        self.assertEqual(zh_br.judges[0].name, 'Adrian Bergmann')
        self.assertEqual(zh_br.judges[0].gender.value, 'male')
        self.assertEqual(zh_br.judges[1].name, 'Stefano Terzi')
        self.assertEqual(zh_br.judges[1].gender.value, 'male')
        self.assertEqual(zh_br.judges[2].name, 'Marlen Patt')
        self.assertEqual(zh_br.judges[2].gender.value, 'male')
        self.assertEqual(zh_br.clerks[0].name, 'Daniel Schweikert')
        self.assertEqual(zh_br.clerks[0].gender.value, 'male')

    def test_ZH_Obergericht(self):
        ZH_Obergericht_test_string = ' '.join(map(str, ZH_Obergericht_test_header))
        sections = {}
        sections[Section.HEADER] = ZH_Obergericht_test_string
        zh_og = c.ZH_Obergericht(sections, namespace_de)
        self.assertIsNone(zh_og.president)
        self.assertEqual(zh_og.judges[0].name, 'Th. Kläusli')
        self.assertEqual(zh_og.judges[0].gender.value, 'male')
        self.assertEqual(zh_og.judges[1].name, 'K. Vogel')
        self.assertEqual(zh_og.judges[1].gender.value, 'male')
        self.assertEqual(zh_og.judges[2].name, 'D. Brugger')
        self.assertEqual(zh_og.judges[2].gender.value, 'male')
        self.assertEqual(zh_og.clerks[0].name, 'Ch. Reitze')
        self.assertEqual(zh_og.clerks[0].gender.value, 'male')

    def test_ZH_Verwaltungsgericht(self):
        ZH_Verwaltungsgericht_test_string = ' '.join(map(str, ZH_Verwaltungsgericht_test_header))
        sections = {}
        sections[Section.HEADER] = ZH_Verwaltungsgericht_test_string
        zh_vg= c.ZH_Verwaltungsgericht(sections, namespace_de)
        self.assertEqual(zh_vg.president.name, 'Tamara Nüssle')
        self.assertEqual(zh_vg.president.gender.value, 'female')
        self.assertEqual(zh_vg.judges[0].name, 'Tamara Nüssle')
        self.assertEqual(zh_vg.judges[0].gender.value, 'female')
        self.assertEqual(zh_vg.judges[1].name, 'Reto Häggi Furrer')
        self.assertEqual(zh_vg.judges[1].gender.value, 'male')
        self.assertEqual(zh_vg.judges[2].name, 'Martin Bertschi')
        self.assertEqual(zh_vg.judges[2].gender.value, 'male')
        self.assertEqual(zh_vg.clerks[0].name, 'David Henseler')
        self.assertEqual(zh_vg.clerks[0].gender.value, 'male')

    def test_ZH_Sozialversicherungsgericht(self):
        ZH_Sozialversicherungsgericht_test_string = ' '.join(map(str, ZH_Sozialversicherungsgericht_test_header))
        sections = {}
        sections[Section.HEADER] = ZH_Sozialversicherungsgericht_test_string
        zh_svg = c.ZH_Sozialversicherungsgericht(sections, namespace_de)
        self.assertEqual(zh_svg.president.name, 'Mosimann')
        self.assertEqual(zh_svg.president.gender.value, 'male')
        self.assertEqual(zh_svg.judges[0].name, 'Mosimann')
        self.assertEqual(zh_svg.judges[0].gender.value, 'male')
        self.assertEqual(zh_svg.judges[1].name, 'Käch')
        self.assertEqual(zh_svg.judges[1].gender.value, 'female')
        self.assertEqual(zh_svg.judges[2].name, 'Sager')
        self.assertEqual(zh_svg.judges[2].gender.value, 'female')
        self.assertEqual(zh_svg.clerks[0].name, 'Kudelski')
        self.assertEqual(zh_svg.clerks[0].gender.value, 'female')


class test_procedural_participation_extracting_functions(unittest.TestCase):
    """
    This function tests whether the procedural participation extracting functions give the correct procedural participation for a given input. If the output is incorrect, an error is shown. 
    """

    def test_ZG_Verwaltungsgericht(self):
        ZG_Verwaltungsgericht_test_string = ', '.join(map(str, ZG_Verwaltungsgericht_test_header))
        zg_vg_json = p.ZG_Verwaltungsgericht(ZG_Verwaltungsgericht_test_string, namespace_de)
        zg_vg = json.loads(zg_vg_json)
        self.assertEqual(zg_vg['plaintiffs'][0]['legal_counsel'][0]['name'], 'B._ AG')
        self.assertEqual(zg_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity')

    def test_ZH_Steuerrekurs(self):
        ZH_Steuerrekurs_test_string = ', '.join(map(str, ZH_Steuerrekurs_test_header))
        zh_sr_json = p.ZH_Steuerrekurs(ZH_Steuerrekurs_test_string, namespace_de)
        zh_sr = json.loads(zh_sr_json)
        self.assertEqual(zh_sr['defendants'][0]['legal_counsel'][0]['name'], 'Steueramt')
        self.assertEqual(zh_sr['defendants'][0]['legal_counsel'][0]['legal_type'], 'legal entity')
        self.assertEqual(zh_sr['plaintiffs'][0]['legal_counsel'][0]['name'], 'C AG')
        self.assertEqual(zh_sr['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'legal entity')

    def test_ZH_Baurekurs(self):
        ZH_Baurekurs_test_string = ', '.join(map(str, ZH_Baurekurs_test_header_2))
        zh_br_json = p.ZH_Baurekurs(ZH_Baurekurs_test_string, namespace_de) 
        zh_br = json.loads(zh_br_json)
        self.assertEqual(zh_br['plaintiffs'][0]['legal_counsel'][0]['name'], 'Dr. iur. Christof Truniger')
        self.assertEqual(zh_br['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person')
        self.assertEqual(zh_br['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male')

    def test_ZH_Obergericht(self):
        ZH_Obergericht_test_string = ', '.join(map(str, ZH_Obergericht_test_header_2))
        zh_og_json = p.ZH_Obergericht(ZH_Obergericht_test_string, namespace_de)
        zh_og = json.loads(zh_og_json)
        self.assertEqual(zh_og['plaintiffs'][0]['legal_counsel'][0]['name'], 'Dr. iur. C. D.')
        self.assertEqual(zh_og['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person')
        self.assertEqual(zh_og['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male')
        self.assertEqual(zh_og['defendants'][0]['legal_counsel'][0]['name'], 'lic. iur. E. F.')
        self.assertEqual(zh_og['defendants'][0]['legal_counsel'][0]['legal_type'], 'natural person')
        self.assertEqual(zh_og['defendants'][0]['legal_counsel'][0]['gender'], 'female')

    def test_ZH_Verwaltungsgericht(self):
        ZH_Verwaltungsgericht_test_string = ', '.join(map(str, ZH_Verwaltungsgericht_test_header))
        zh_vg_json = p.ZH_Verwaltungsgericht(ZH_Verwaltungsgericht_test_string, namespace_de)
        zh_vg = json.loads(zh_vg_json)
        self.assertEqual(zh_vg['plaintiffs'][0]['legal_counsel'][0]['name'], 'B')
        self.assertEqual(zh_vg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person')
        self.assertEqual(zh_vg['plaintiffs'][0]['legal_counsel'][0]['gender'], 'unknown')

    def test_ZH_Sozialversicherungsgericht(self):
        ZH_Sozialversicherungsgericht_test_string = ', '.join(map(str, ZH_Sozialversicherungsgericht_test_header))
        zh_svg_json = p.ZH_Sozialversicherungsgericht(ZH_Sozialversicherungsgericht_test_string, namespace_de)
        zh_svg = json.loads(zh_svg_json)
        self.assertEqual(zh_svg['plaintiffs'][0]['legal_counsel'][0]['name'], 'Dr. Kreso Glavas')
        self.assertEqual(zh_svg['plaintiffs'][0]['legal_counsel'][0]['legal_type'], 'natural person')
        self.assertEqual(zh_svg['plaintiffs'][0]['legal_counsel'][0]['gender'], 'male')


if __name__ == '__main__':
    unittest.main()
