import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, FormField, Header, Pagination, SelectCSD, Table, Toggle } from 'components';

import { DEFAULT_TABLE_PAGE_SIZE } from 'consts';
import { useBreadcrumbs, useCollection } from 'hooks';
import { ROUTES } from 'routes';
import { useLazyGetPoolsInstancesQuery } from 'services/pool';

import { useColumnsDefinitions, useEmptyMessages, useFilters } from './hooks';

import styles from './styles.module.scss';

export const FleetList: React.FC = () => {
    const { t } = useTranslation();
    const [data, setData] = useState<IInstanceListItem[]>([]);
    const [pagesCount, setPagesCount] = useState<number>(1);
    const [disabledNext, setDisabledNext] = useState(false);

    useBreadcrumbs([
        {
            text: t('navigation.fleets'),
            href: ROUTES.FLEETS.LIST,
        },
    ]);

    const {
        onlyActive,
        setOnlyActive,
        isDisabledClearFilter,
        clearFilters,
        projectOptions,
        selectedProject,
        setSelectedProject,
    } = useFilters();

    const [getPools, { isLoading, isFetching }] = useLazyGetPoolsInstancesQuery();
    const isDisabledPagination = isLoading || isFetching || data.length === 0;

    const getPoolsRequest = (params?: Partial<TPoolInstancesRequestParams>) => {
        return getPools({
            only_active: onlyActive,
            project_name: selectedProject?.value,
            limit: DEFAULT_TABLE_PAGE_SIZE,
            ...params,
        }).unwrap();
    };

    useEffect(() => {
        getPoolsRequest().then((result) => {
            setPagesCount(1);
            setDisabledNext(false);
            setData(result);
        });
    }, [onlyActive, selectedProject?.value]);

    const { columns } = useColumnsDefinitions();
    const { renderEmptyMessage, renderNoMatchMessage } = useEmptyMessages({ clearFilters, isDisabledClearFilter });

    const nextPage = async () => {
        if (data.length === 0 || disabledNext) {
            return;
        }

        try {
            const result = await getPoolsRequest({
                prev_created_at: data[data.length - 1].created,
                prev_id: data[data.length - 1].id,
            });

            if (result.length > 0) {
                setPagesCount((count) => count + 1);
                setData(result);
            } else {
                setDisabledNext(true);
            }
        } catch (e) {
            console.log(e);
        }
    };

    const prevPage = async () => {
        if (pagesCount === 1) {
            return;
        }

        try {
            const result = await getPoolsRequest({
                prev_created_at: data[0].created,
                prev_id: data[0].id,
                ascending: true,
            });

            setDisabledNext(false);

            if (result.length > 0) {
                setPagesCount((count) => count - 1);
                setData(result);
            } else {
                setPagesCount(1);
            }
        } catch (e) {
            console.log(e);
        }
    };

    const { items, collectionProps } = useCollection<IInstanceListItem>(data, {
        filtering: {
            empty: renderEmptyMessage(),
            noMatch: renderNoMatchMessage(),
        },
        pagination: { pageSize: 20 },
        selection: {},
    });

    const renderCounter = () => {
        if (!data?.length) return '';

        return `(${data.length})`;
    };

    return (
        <Table
            {...collectionProps}
            variant="full-page"
            columnDefinitions={columns}
            items={items}
            loading={isLoading || isFetching}
            loadingText={t('common.loading')}
            stickyHeader={true}
            header={
                <Header variant="awsui-h1-sticky" counter={renderCounter()}>
                    {t('navigation.fleets')}
                </Header>
            }
            filter={
                <div className={styles.filters}>
                    <div className={styles.select}>
                        <FormField label={t('projects.run.project')}>
                            <SelectCSD
                                disabled={!projectOptions?.length}
                                options={projectOptions}
                                selectedOption={selectedProject}
                                onChange={(event) => {
                                    setSelectedProject(event.detail.selectedOption);
                                }}
                                placeholder={t('projects.run.project_placeholder')}
                                expandToViewport={true}
                                filteringType="auto"
                            />
                        </FormField>
                    </div>

                    <div className={styles.activeOnly}>
                        <Toggle onChange={({ detail }) => setOnlyActive(detail.checked)} checked={onlyActive}>
                            {t('fleets.active_only')}
                        </Toggle>
                    </div>

                    <div className={styles.clear}>
                        <Button formAction="none" onClick={clearFilters} disabled={isDisabledClearFilter}>
                            {t('common.clearFilter')}
                        </Button>
                    </div>
                </div>
            }
            pagination={
                <Pagination
                    currentPageIndex={pagesCount}
                    pagesCount={pagesCount}
                    openEnd={!disabledNext}
                    disabled={isDisabledPagination}
                    onPreviousPageClick={prevPage}
                    onNextPageClick={nextPage}
                />
            }
        />
    );
};
