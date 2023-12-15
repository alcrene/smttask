"""
Unit tests for the parameters.validators module

Also see the doctests in doc/validation.txt
"""

import parameters
from parameters.random import GammaDist, UniformDist, NormalDist, ParameterDist
import os
import sys
import unittest
import types
from copy import deepcopy
import pickle

import parameters.validators
from parameters.validators import congruent_dicts
from parameters import *

import yaml


ps = ParameterSet({'a':1, 'b':2}, label="PS1")
ps2 = ParameterSet({'ps':ps, 'c':19}, label="PS2")
ps3 = ParameterSet({'hello': 'world', 'ps2': ps2, 'null': None,
                    'true': False, 'mylist': [1,2,3,4],
                    'mydict': {'c': 3, 'd':4}, 'yourlist': [1,2,{'e':5, 'f':6}],
                    }, label="PS3")

schema3 = ParameterSchema({'mylist': parameters.validators.Subclass(type=list),
                           'true': parameters.validators.Subclass(type=bool),
                           'yourlist': parameters.validators.Subclass(type=list),
                           'ps2': {'ps': {'a':parameters.validators.Subclass(type=int),
                                          'b':parameters.validators.Subclass(type=int)},
                                   'c': parameters.validators.Subclass(type=int)},
                           'null': parameters.validators.Subclass(type=type(None)),
                           'mydict': {'c': parameters.validators.Subclass(type=int),
                                      'd': parameters.validators.Subclass(type=int)},
                           'hello': parameters.validators.Subclass(type=str)})


class ParameterSchemeTest(unittest.TestCase):

    def test_simple_create(self):
        s1 = ParameterSchema({'age': 0, 'height': Subclass(float)})
        # is equivalent to
        s2 = ParameterSchema({'age': Subclass(int), 'height': Subclass(float)})


        assert s1 == s2


    def test_validate_generated_schema(self):
        s1 = ParameterSchema(ps3)

        v = CongruencyValidator()
        assert v.validate(ps3,s1)==True

    def test_eval(self):

        s1 = ParameterSchema(ps3)

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        s1.mylist = parameters.validators.Eval('isinstance(x,list)',var='x')
        assert v.validate(ps3,s1)==True

        # test default var 'leaf'
        s1.hello = parameters.validators.Eval('isinstance(leaf,str)')
        assert v.validate(ps3,s1)==True

        # test failure and var as non-kwarg
        s1.mylist = parameters.validators.Eval('isinstance(x,str)','x')

        r = False
        try:
            r = v.validate(ps3,s1)
        except Exception as e:
            assert isinstance(e,ValidationError)
            assert e.path=='mylist'
            assert e.parameter==ps3.mylist
            assert e.schema_base==s1.mylist
        assert r == False

    def test_complex_eval(self):

        p_str = """
        list: [1,2,3,4]
        flist: [1.0,2.0,3.0,4.0]
        """

        s_str = """
        list: !!python/object:parameters.validators.Eval { expr: 'isinstance(x,list) and all([isinstance(elem,int) for elem in x])', var: 'x'}
        flist: !!python/object:parameters.validators.Eval { expr: 'isinstance(x,list) and all([isinstance(elem,float) for elem in x])', var: 'x'}
        """


        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        #s1.mylist = parameters.validators.Eval('isinstance(x,list) and all([isinstance(elem,int) for elem in x])',var='x')
        assert v.validate(p1,s1)==True

    def test_complex_dict_eval(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: ["TCPROJ_POmAnt_to_SOM_S1_", 469.0]
        """

        s_str = """
        item: !!python/object:parameters.validators.Eval { expr: 'isinstance(x,dict) and all', var: 'x'}
        """


        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        #s1.mylist = parameters.validators.Eval('isinstance(x,list) and all([isinstance(elem,int) for elem in x])',var='x')
        assert v.validate(p1,s1, strict_tree=False)==True

    def test_wildcard_fail_eval(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: 1.0
        """

        s_str = """
        item:
            <<ANY_KEY>>: ["MORPH_BASE_NAME", 1.0]
        """

        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        self.assertRaises(ValidationError, v.validate, p1, s1, strict_tree=False)

    def test_wildcard_exception_eval(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: 1.0
            PROJ_VPM2: ["TCPROJ_VPM_to_SOM_S1_", 968.35]

        """

        s_str = """
        item:
            <<ANY_KEY>>: ["MORPH_BASE_NAME", 1.0]
            PROJ_POM: 1.0
        """

        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        assert v.validate(p1,s1, strict_tree=False)==True


    def test_wildcard_eval(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: ["TCPROJ_POmAnt_to_SOM_S1_", 469.0]
        """

        s_str = """
        item:
            <<ANY_KEY>>: ["MORPH_BASE_NAME", 1.0]
        """

        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        #s1.mylist = parameters.validators.Eval('isinstance(x,list) and all([isinstance(elem,int) for elem in x])',var='x')
        assert v.validate(p1,s1, strict_tree=False)==True

    def test_wildcard_lax_list_force_fail(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: ["TCPROJ_POmAnt_to_SOM_S1_", 123.0]
            PROJ_FAIL: [""]
        """

        s_str = """
        item:
            <<ANY_KEY>>: !!python/object:parameters.validators.StrictContentTypes {val: ["",100.0] }
        """

        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        #s1.mylist = parameters.validators.Eval('isinstance(x,list) and all([isinstance(elem,int) for elem in x])',var='x')
        self.assertRaises(ValidationError, v.validate, p1, s1, strict_tree=False)

    def test_wildcard_lax_list(self):

        p_str = """
        item:
            # PROJ_NAME: ["morphology base name", density(#/mm^2)]
            PROJ_VPM: ["TCPROJ_VPM_to_SOM_S1_", 968.35]
            PROJ_POM: ["TCPROJ_POmAnt_to_SOM_S1_", 469.0]
            PROJ_WANT_FAILING_BUT_DOESNT: [""]
        """

        s_str = """
        item:
            <<ANY_KEY>>: ["MORPH_BASE_NAME", 1.0]
        """

        s1 = ParameterSchema(yaml.full_load(s_str))
        p1 = ParameterSet(yaml.full_load(p_str))

        v = CongruencyValidator()
        # test Eval which checks for isinstance(x,list)
        #s1.mylist = parameters.validators.Eval('isinstance(x,list) and all([isinstance(elem,int) for elem in x])',var='x')
        assert v.validate(p1,s1, strict_tree=False)==True



    def test_simple_validate(self):
        v = CongruencyValidator()
        assert v.validate(ps3,schema3)==True

    def test_validation_failure_info(self):
        """ Test the error output for an failed validation against a schema"""

        s1 = ParameterSchema(ps3)

        s1.ps2.ps.a = Subclass(type=float)
        v = CongruencyValidator()
        r = False
        try:
            r = v.validate(ps3,s1)
        except Exception as e:
            assert isinstance(e,ValidationError)
            assert e.path=='ps2.ps.a'
            assert e.parameter==ps3.ps2.ps.a
            assert e.schema_base==s1.ps2.ps.a
        assert r == False


    def test_yaml_schema(self):
        import yaml

        conf1_str = """
        # user info
        username: joe
        email:  joe@example.com

        # recipes
        recipes:
           all: /somewhere1/path1.xml
           specific: /somewhere2/path2.xml
        numbers:
           float: 1.0
        """

        schema_str = """
        # user info
        username: ''
        email:  ''

        # recipes
        recipes:
           all: ''
           specific: ''
        numbers:
           float: !!python/object:parameters.validators.Subclass { type: !!python/name:float }
        """

        schema = ParameterSchema(yaml.full_load(schema_str))
        conf = ParameterSet(yaml.full_load(conf1_str))

        v = CongruencyValidator()
        assert v.validate(conf,schema)

        del conf.recipes['all']
        r = False
        try:
            r = v.validate(conf,schema)
        except Exception as e:
            assert isinstance(e,ValidationError)
            assert e.path=='recipes.all'
            assert e.parameter=='<MISSING>'
            assert e.schema_base==schema.recipes.all

        assert r == False


    def test_yaml_file_schema(self):
        import yaml, tempfile

        conf1_str = """
        # user info
        username: joe
        email:  joe@example.com

        # recipes
        recipes:
           all: /somewhere1/path1.xml
           specific: /somewhere2/path2.xml
        numbers:
           float: 1.0
        """

        schema_str = """
        # user info
        username: ''
        email: ''

        # recipes
        recipes:
           all: ''
           specific: ''
        numbers:
           float: !!python/object:parameters.validators.Subclass { type: !!python/name:float }
        """

        def write_to_yaml_tf(s):
            tf = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w')
            tf.writelines(s)
            tf.flush()
            tf.seek(0)
            return tf

        schema_tf = write_to_yaml_tf(schema_str)
        conf1_tf = write_to_yaml_tf(conf1_str)

        with schema_tf:
            with conf1_tf:

                schema = ParameterSchema(schema_tf.name)
                conf = ParameterSet(conf1_tf.name)

                v = CongruencyValidator()
                assert v.validate(conf,schema)

                del conf.recipes['all']
                r = False
                try:
                    r = v.validate(conf,schema)
                except Exception as e:
                    assert isinstance(e,ValidationError)
                    assert e.path=='recipes.all'
                    assert e.parameter=='<MISSING>'
                    assert e.schema_base==schema.recipes.all

                assert r == False


    def test_congruent_dicts(self):
        assert congruent_dicts({},{})
        assert congruent_dicts(None,1)
        assert congruent_dicts({},{'hello':10})==False
        assert congruent_dicts({'hello':False},{'hello':10})
        assert congruent_dicts({'hello':{'a':False}},{'hello':{'a':10}})
        assert congruent_dicts({'hello':{'a':False}},{'hello':{'b':10}})==False

        conf1_str = """
        # user info
        username: joe
        email:  joe@example.com

        # executables
        builder_path: /bgscratch/bbp/bin/sgi/
        detector_path: /bgscratch/bbp/bin/sgi/

        # recipes
        recipes:
           all: /somewhere/builderRecipeAllPathways.xml
           specific: /somewhere/builderRecipeSpecificPathways.xml
        """

        conf2_str = """
        # user info
        username: null
        email:  null

        # executables
        builder_path: null
        detector_path: null

        # recipes
        recipes:
           all: null
           specific: null
        """

        conf3_str = """

        # executables
        builder_path: null
        detector_path: null

        # recipes
        recipes:
           all: null
        """

        conf1 = yaml.full_load(conf1_str)
        conf2 = yaml.full_load(conf2_str)
        conf3 = yaml.full_load(conf3_str)

        assert congruent_dicts(conf1,conf2)
        # conf3 is subset of conf1 heirarchy, should return false
        assert congruent_dicts(conf1,conf3)==False

        # check subset==True
        assert congruent_dicts(conf1,conf3,subset=True)

        # check that additional field not in conf1 returns False
        conf3['recipes']['something'] = 1

        assert congruent_dicts(conf1,conf3,subset=True)==False




if __name__ == '__main__':
    unittest.main()
